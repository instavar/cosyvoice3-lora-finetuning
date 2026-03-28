#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'third_party' / 'Matcha-TTS'))

import inflect
import numpy as np
import requests
import torch
import torchaudio

import ttsfrd
from wetext import Normalizer as WeTextNormalizer

from cosyvoice.cli.cosyvoice import CosyVoice3
from cosyvoice.utils.frontend_utils import (
    contains_chinese,
    is_only_punctuation,
    remove_bracket,
    replace_blank,
    replace_corner_mark,
    spell_out_number,
    split_paragraph,
)


DEFAULT_PROMPT_WAV = ''
DEFAULT_PROMPT_TEXT = ''
DEFAULT_PRETRAINED_DIR = str(REPO_ROOT / 'pretrained_models' / 'Fun-CosyVoice3-0.5B')
CHINESE_MONEY_RE = re.compile(r'(?P<num>\d[\d,]*)(?P<unit>新币|新幣|人民币|人民幣|港币|港幣|美元|美金|元|块)')
ENGLISH_ABBREVIATION_RE = re.compile(
    r"\b(?:[A-Z]{2,}(?:\s+[A-Z]{2,})*|Dr|Mr|Mrs|Ms|No|St|Rd|Ave|Blvd|P\.M\.|A\.M\.|p\.m\.|a\.m\.)\b"
)
MIXED_ACRONYM_RE = re.compile(r'\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\b')


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_texts(text: str | None, texts_file: str | None) -> list[str]:
    if text:
        return [text]
    if not texts_file:
        raise ValueError('Provide --text or --texts-file')
    path = Path(texts_file)
    if not path.exists():
        raise FileNotFoundError(f'Texts file not found: {texts_file}')
    lines: list[str] = []
    with path.open('r', encoding='utf-8') as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            lines.append(line)
    if not lines:
        raise ValueError(f'No valid lines in {texts_file}')
    return lines


def int_to_chinese(num: int) -> str:
    if num == 0:
        return '零'
    digits = '零一二三四五六七八九'
    units = ['', '十', '百', '千']
    big_units = ['', '万', '亿']

    def four_to_chinese(chunk: int) -> str:
        result: list[str] = []
        zero_pending = False
        for idx, digit in enumerate(f'{chunk:04d}'):
            value = int(digit)
            pos = 3 - idx
            if value == 0:
                zero_pending = len(result) > 0
                continue
            if zero_pending:
                result.append('零')
                zero_pending = False
            if not (value == 1 and pos == 1 and not result):
                result.append(digits[value])
            result.append(units[pos])
        return ''.join(result).rstrip('零')

    parts: list[str] = []
    group_idx = 0
    need_zero = False
    while num > 0:
        chunk = num % 10000
        if chunk:
            chunk_text = four_to_chinese(chunk)
            if big_units[group_idx]:
                chunk_text += big_units[group_idx]
            if need_zero:
                parts.append('零')
                need_zero = False
            parts.append(chunk_text)
        else:
            if parts:
                need_zero = True
        num //= 10000
        group_idx += 1
    return ''.join(reversed(parts)).replace('一十', '十', 1)


def normalize_chinese_money(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        num_raw = match.group('num').replace(',', '')
        if not num_raw.isdigit():
            return match.group(0)
        return f'{int_to_chinese(int(num_raw))}{match.group("unit")}'

    return CHINESE_MONEY_RE.sub(repl, text)


class HybridTextNormalizer:
    def __init__(self, cosyvoice: CosyVoice3) -> None:
        self.tokenizer = cosyvoice.frontend.tokenizer
        self.allowed_special = cosyvoice.frontend.allowed_special
        self.inflect_parser = inflect.engine()
        self.ttsfrd_engine = ttsfrd.TtsFrontendEngine()
        resource_dir = REPO_ROOT / 'pretrained_models' / 'CosyVoice-ttsfrd' / 'resource'
        if not resource_dir.exists():
            raise FileNotFoundError(f'ttsfrd resource not found: {resource_dir}')
        if self.ttsfrd_engine.initialize(str(resource_dir)) is not True:
            raise RuntimeError('failed to initialize ttsfrd resource')
        self.ttsfrd_engine.set_lang_type('pinyinvg')
        self.zh_tn_model = self._build_wetext_normalizer(remove_erhua=False)
        self.en_tn_model = self._build_wetext_normalizer()

    def _build_wetext_normalizer(self, remove_erhua: bool | None = None) -> WeTextNormalizer:
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                kwargs = {}
                if remove_erhua is not None:
                    kwargs['remove_erhua'] = remove_erhua
                return WeTextNormalizer(**kwargs)
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                if attempt == 2:
                    break
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f'Failed to initialize wetext normalizer after retries: {last_exc}') from last_exc

    def route_policy(self, text: str, requested: str) -> str:
        if requested in {'ttsfrd', 'wetext'}:
            return requested
        if contains_chinese(text):
            if MIXED_ACRONYM_RE.search(text):
                return 'wetext'
            return 'wetext'
        if ENGLISH_ABBREVIATION_RE.search(text):
            return 'ttsfrd'
        return 'wetext'

    def pre_normalize(self, text: str) -> str:
        if contains_chinese(text):
            return normalize_chinese_money(text)
        return text

    def _split_and_clean(self, text: str, lang: str) -> list[str]:
        texts = list(
            split_paragraph(
                text,
                lambda value: self.tokenizer.encode(value, allowed_special=self.allowed_special),
                lang,
                token_max_n=80,
                token_min_n=60,
                merge_len=20,
                comma_split=False,
            )
        )
        return [item for item in texts if not is_only_punctuation(item)]

    def normalize_with_ttsfrd(self, text: str) -> list[str]:
        payload = json.loads(self.ttsfrd_engine.do_voicegen_frd(text))
        texts = [item['text'] for item in payload['sentences']]
        return [item for item in texts if not is_only_punctuation(item)]

    def normalize_with_wetext(self, text: str) -> list[str]:
        if contains_chinese(text):
            text = self.zh_tn_model.normalize(text)
            text = text.replace('\n', '')
            text = replace_blank(text)
            text = replace_corner_mark(text)
            text = text.replace('.', '。')
            text = text.replace(' - ', '，')
            text = remove_bracket(text)
            text = re.sub(r'[，,、]+$', '。', text)
            return self._split_and_clean(text, 'zh')
        text = self.en_tn_model.normalize(text)
        text = spell_out_number(text, self.inflect_parser)
        return self._split_and_clean(text, 'en')

    def normalize(self, text: str, requested_policy: str) -> dict[str, object]:
        pre_normalized = self.pre_normalize(text.strip())
        selected = self.route_policy(pre_normalized, requested_policy)
        if selected == 'ttsfrd':
            segments = self.normalize_with_ttsfrd(pre_normalized)
        else:
            segments = self.normalize_with_wetext(pre_normalized)
        if not segments:
            raise RuntimeError('No normalized text segments produced')
        return {
            'frontend_selected': selected,
            'pre_normalized_text': pre_normalized,
            'normalized_segments': segments,
        }


def save_metadata(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def resolve_repo_relative(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description='CosyVoice3 hybrid-normalization inference helper.')
    parser.add_argument('--pretrained-dir', default=DEFAULT_PRETRAINED_DIR, help='CosyVoice3 pretrained model directory')
    parser.add_argument('--prompt-wav', required=True, help='Prompt wav path')
    parser.add_argument('--prompt-text', required=True, help='Prompt text matching the prompt audio')
    parser.add_argument('--text', default='', help='Single input text')
    parser.add_argument('--texts-file', default='', help='Text file with one sentence per line')
    parser.add_argument('--out-wav', default='', help='Output wav path (single text only)')
    parser.add_argument('--out-dir', default='', help='Output directory (multiple texts)')
    parser.add_argument('--speed', type=float, default=1.0, help='Speech speed multiplier')
    parser.add_argument('--fp16', action='store_true', help='Enable fp16 inference')
    parser.add_argument('--frontend-policy', choices=['auto', 'ttsfrd', 'wetext'], default='auto', help='Frontend routing policy')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--dry-run', action='store_true', help='Only emit normalization metadata; do not generate audio')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    seed_everything(args.seed)

    texts = read_texts(args.text or None, args.texts_file or None)
    if args.out_wav and len(texts) != 1:
        raise ValueError('--out-wav only supports a single --text')
    if not args.dry_run and not args.out_wav and not args.out_dir:
        raise ValueError('Provide --out-wav or --out-dir unless --dry-run is used')

    prompt_wav = Path(args.prompt_wav)
    if not prompt_wav.exists():
        raise FileNotFoundError(f'Prompt wav not found: {prompt_wav}')

    pretrained_dir = resolve_repo_relative(args.pretrained_dir)
    cosyvoice = CosyVoice3(pretrained_dir, fp16=args.fp16)
    normalizer = HybridTextNormalizer(cosyvoice)

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, object]] = []
    with torch.no_grad():
        for idx, raw_text in enumerate(texts, start=1):
            normalized = normalizer.normalize(raw_text, args.frontend_policy)
            run_record: dict[str, object] = {
                'index': idx,
                'raw_text': raw_text,
                'frontend_policy': args.frontend_policy,
                **normalized,
            }

            if not args.dry_run:
                chunks = []
                for segment in normalized['normalized_segments']:
                    for model_output in cosyvoice.inference_zero_shot(
                        str(segment),
                        args.prompt_text,
                        str(prompt_wav),
                        stream=False,
                        speed=args.speed,
                        text_frontend=False,
                    ):
                        chunks.append(model_output['tts_speech'].cpu())
                if not chunks:
                    raise RuntimeError(f'No audio generated for text: {raw_text}')
                speech = torch.cat(chunks, dim=1).to(torch.float32)
                if args.out_wav:
                    out_path = Path(args.out_wav)
                else:
                    out_path = out_dir / f'sample_{idx:02d}.wav'
                torchaudio.save(str(out_path), speech, cosyvoice.sample_rate)
                run_record.update(
                    {
                        'out_path': str(out_path),
                        'sample_rate': cosyvoice.sample_rate,
                        'duration_seconds': round(float(speech.shape[-1] / cosyvoice.sample_rate), 3),
                    }
                )
                logging.info(
                    'Saved %s using %s',
                    out_path,
                    normalized['frontend_selected'],
                )
            runs.append(run_record)

    payload = {
        'pretrained_dir': args.pretrained_dir,
        'prompt_wav': args.prompt_wav,
        'prompt_text': args.prompt_text,
        'speed': args.speed,
        'fp16': args.fp16,
        'seed': args.seed,
        'dry_run': args.dry_run,
        'runs': runs,
    }

    if args.out_wav:
        metadata_path = Path(args.out_wav).with_suffix('.json')
    elif out_dir:
        metadata_path = out_dir / 'metadata.json'
    else:
        metadata_path = Path('/tmp/cosyvoice3_hybrid_metadata.json')
    save_metadata(metadata_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f'metadata={metadata_path}')


if __name__ == '__main__':
    main()
