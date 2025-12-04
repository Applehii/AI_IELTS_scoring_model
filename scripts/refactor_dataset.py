#!/usr/bin/env python3
"""
Refactor dataset sample files to canonical RAG-ready format.
Writes back updated .txt files and produces:
 - data/samples/task_index.json  (metadata for all samples)
 - scripts/changes_log.txt      (list of changed files and brief notes)

This script is conservative: it preserves original content, extracts existing sections
when present, and inserts a JSON metadata header, a [SUMMARY] (1-2 sentences), and
canonical headings: [QUESTION], [SUMMARY], [SAMPLE_ANSWER], [OVERVIEW], [RATIONALE].

It uses filename and in-file markers to infer band and task. It will try not to alter meaning.
"""

import re
import os
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = ROOT / 'data' / 'samples'
CHANGES_LOG = Path(__file__).resolve().parent / 'changes_log.txt'
TASK_INDEX = ROOT / 'data' / 'samples' / 'task_index.json'

# simple stopwords for topic extraction
STOPWORDS = set("""the a an and or of to in for on with at by from about as is are was were be have has had this that these those it its""".split())


def extract_sections(text):
    # Normalize CRLF
    text = text.replace('\r\n','\n').strip()
    lines = text.split('\n')
    joined = '\n'.join(lines)

    # Find header markers case-insensitive and canonicalize common variants
    # We'll map header variants to canonical headings: QUESTION, SUMMARY, SAMPLE_ANSWER, OVERVIEW, RATIONALE
    sections = {'QUESTION':'','SUMMARY':'','SAMPLE_ANSWER':'','OVERVIEW':'','RATIONALE':''}

    # Generic header regex to find bracketed header lines
    header_regex = re.compile(r'^\s*\[(?P<header>[^\]]+)\]\s*$', flags=re.I | re.M)
    headers = [(m.start(), m.end(), m.group('header').strip()) for m in header_regex.finditer(joined)]
    if headers:
        # Append sentinel end
        headers.append((len(joined),len(joined),'END'))
        for i in range(len(headers)-1):
            raw_name = headers[i][2].strip()
            # canonicalize name for matching
            name_up = re.sub(r'[^A-Z0-9 ]', ' ', raw_name.upper())
            name_up = re.sub(r'\s+', ' ', name_up).strip()
            start = headers[i][1]
            end = headers[i+1][0]
            body = joined[start:end].strip()

            # Determine canonical heading
            canonical = None
            if 'QUESTION' in name_up:
                canonical = 'QUESTION'
            elif 'SUMMARY' in name_up:
                canonical = 'SUMMARY'
            elif 'SAMPLE' in name_up or 'ANSWER' in name_up or name_up.startswith('BAND '):
                # headers like '[BAND 4 SAMPLE ANSWER]' or '[SAMPLE ANSWER]' -> SAMPLE_ANSWER
                canonical = 'SAMPLE_ANSWER'
            elif 'OVERV' in name_up or 'OVER' in name_up and 'VIEW' in name_up:
                # catch misspellings like OVERIEW, OVRVIEW etc. Use substring checks
                canonical = 'OVERVIEW'
            elif 'WHY' in name_up or 'RATION' in name_up or 'RATIONALE' in name_up:
                canonical = 'RATIONALE'
            else:
                # not recognized: fold into SAMPLE_ANSWER but do not keep the non-canonical bracketed header label
                canonical = 'SAMPLE_ANSWER'

            # Append body into the canonical section (preserve ordering by concatenation)
            if canonical in ('SAMPLE_ANSWER','RATIONALE'):
                # accumulate multi-block fields
                if sections[canonical]:
                    sections[canonical] += '\n\n' + body
                else:
                    sections[canonical] = body
            else:
                # QUESTION, SUMMARY, OVERVIEW: replace if empty, else append
                if sections[canonical]:
                    sections[canonical] += '\n\n' + body
                else:
                    sections[canonical] = body

        # strip whitespace from each section
        for k in sections:
            sections[k] = sections[k].strip()
        return sections
    else:
        # No explicit headers; attempt heuristics
        # Try to find the question as the first paragraph ending with question mark
        q = ''
        m = re.search(r'^(.*?\?)\s', joined)
        if m:
            q = m.group(1).strip()
        else:
            # fallback: first paragraph
            paras = re.split(r'\n\s*\n', joined)
            q = paras[0].strip() if paras else ''

        # Find band marker like 'Band 4' or 'BAND 4' in text
        # Find overview or why band by keywords 'Overview' or 'Why band' or 'Why'
        # We'll attempt to find the last short paragraph that looks evaluative
        paras = [p.strip() for p in re.split(r'\n\s*\n', joined) if p.strip()]
        sample = ''
        overview = ''
        rationale = ''
        if paras:
            # If the first para starts with 'Band' or contains numbers, treat others as sample
            if re.match(r'(?i)^Band\s*\d', paras[0]):
                sample = '\n\n'.join(paras)
            else:
                sample = '\n\n'.join(paras[1:]) if len(paras)>1 else paras[0]

        sections['QUESTION'] = q
        sections['SAMPLE_ANSWER'] = sample.strip()
        sections['OVERVIEW'] = overview
        sections['RATIONALE'] = rationale
        sections['SUMMARY'] = ''
        return sections


def generate_summary(sections):
    # If existing SUMMARY present, shorten to 1-2 sentences
    s = sections.get('SUMMARY','').strip()
    if s:
        # take first 2 sentences
        parts = re.split(r'(?<=[.!?])\s+', s)
        return ' '.join(parts[:2]).strip()
    # else derive from SAMPLE_ANSWER or QUESTION
    # Derive summary strictly from SAMPLE_ANSWER first, else QUESTION. Do not invent facts.
    base = sections.get('SAMPLE_ANSWER','').strip() or sections.get('QUESTION','').strip()
    if not base:
        return ''
    # extract first sentence
    sentences = re.split(r'(?<=[.!?])\s+', base)
    if sentences:
        # take at most two sentences and ensure concise
        first = sentences[0].strip()
        second = sentences[1].strip() if len(sentences) > 1 else ''
        # truncate sentences to avoid accidental invention or long tails
        def trim_sent(s, max_words=35):
            words = s.split()
            return ' '.join(words[:max_words]) + ('...' if len(words) > max_words else '')
        first = trim_sent(first, max_words=40)
        if second:
            second = trim_sent(second, max_words=40)
            return (first + ' ' + second).strip()
        return first
    return ''


def infer_band_from_filename(name):
    m = re.search(r'band(\d)', name, flags=re.I)
    if m:
        return int(m.group(1))
    m2 = re.search(r'Band\s*(\d)', name, flags=re.I)
    if m2:
        return int(m2.group(1))
    return None


def extract_topics(text, limit=6):
    # naive keyword extractor: split words, filter stopwords and short words, pick most frequent
    words = re.findall(r"[A-Za-z]{3,}", text.lower())
    freq = {}
    for w in words:
        if w in STOPWORDS:
            continue
        freq[w] = freq.get(w,0)+1
    # sort by frequency then by length
    keys = sorted(freq.items(), key=lambda kv:(-kv[1], -len(kv[0])))
    topics = [k for k,v in keys][:limit]
    return topics


def normalize_band_label(band, sample_answer):
    # Ensure first line of SAMPLE_ANSWER starts with 'Band X' (preserve existing if present)
    sa = sample_answer.strip()
    if re.match(r'(?i)^Band\s*\d', sa):
        return sa
    else:
        label = f"Band {band}\n" if band else ''
        return (label + sa).strip()


def process_file(path: Path):
    raw = path.read_text(encoding='utf-8')
    sections = extract_sections(raw)
    # infer band from filename or sample content
    band = infer_band_from_filename(path.name)
    if not band:
        m = re.search(r'Band\s*(\d)', sections.get('SAMPLE_ANSWER',''), flags=re.I)
        if m:
            try:
                band = int(m.group(1))
            except:
                band = None

    # task type from path: e.g., task1/tables -> task1_tables
    parts = list(path.parts)
    # find 'writing_samples' index
    task_type = None
    if 'writing_samples' in parts:
        idx = parts.index('writing_samples')
        tail = parts[idx+1: idx+3]  # e.g., ['task1','tables']
        task_type = '_'.join(tail) if tail else 'unknown'
    else:
        # fallback use parent folder names
        task_type = path.parent.name

    sample_id = path.stem

    # ensure canonical sections
    question = sections.get('QUESTION','').strip()
    if not question:
        # try to extract from raw first paragraph
        para = re.split(r'\n\s*\n', raw)
        question = para[0].strip() if para else ''

    sample_answer = sections.get('SAMPLE_ANSWER','').strip()
    overview = sections.get('OVERVIEW','').strip()
    rationale = sections.get('RATIONALE','').strip()

    # build summary
    summary = generate_summary(sections)

    # topics: use question + first 200 chars of sample_answer
    topic_src = (question + ' ' + sample_answer[:400]).strip()
    topics = extract_topics(topic_src)

    # Build metadata with deterministic key order
    metadata = {
        'band': band if band else None,
        'task': task_type,
        'sample_id': sample_id,
        'topics': topics,
        'rag_ready': True
    }

    # Normalize sample_answer band label
    sample_answer_norm = normalize_band_label(band, sample_answer)

    # Ensure defaults for missing OVERVIEW / RATIONALE per strict spec
    if not overview:
        overview = 'No overview provided.'
    if not rationale:
        rationale = 'No rationale provided.'

    # Force SAMPLE_ANSWER to start with 'Band X' exactly when band is known
    if band:
        sample_answer_norm = normalize_band_label(band, sample_answer)
    else:
        # if band unknown, keep sample_answer as-is but do not add Band label
        sample_answer_norm = sample_answer.strip()

    # Build canonical content with exact ordering and blank lines between sections
    # JSON header: compact single-line with keys in the same order
    json_header = json.dumps(metadata, ensure_ascii=False, separators=(',', ': '))
    parts = [json_header, '', '[QUESTION]', question or '', '', '[SUMMARY]', summary or '', '', '[SAMPLE_ANSWER]', sample_answer_norm or '', '', '[OVERVIEW]', overview, '', '[RATIONALE]', rationale, '']
    canonical_text = '\n'.join(parts)

    # Always write back to ensure strict normalization; count as changed only if content differs
    changed = canonical_text.strip() != raw.strip()
    # Write back unconditionally to guarantee normalization across all files
    path.write_text(canonical_text, encoding='utf-8')
    return metadata, changed


def main():
    all_metadata = []
    changed_files = []
    total = 0
    for dirpath, dirnames, filenames in os.walk(SAMPLES_DIR):
        for fname in filenames:
            if not fname.endswith('.txt'):
                continue
            # skip index file if any
            if fname == 'task_index.json':
                continue
            p = Path(dirpath) / fname
            try:
                meta, changed = process_file(p)
                if meta:
                    all_metadata.append({**meta, 'path': str(p.relative_to(ROOT))})
                if changed:
                    changed_files.append(str(p.relative_to(ROOT)))
                total += 1
            except Exception as e:
                print(f"Error processing {p}: {e}")

    # write task_index.json
    TASK_INDEX.parent.mkdir(parents=True, exist_ok=True)
    TASK_INDEX.write_text(json.dumps(all_metadata, ensure_ascii=False, indent=2), encoding='utf-8')

    # write changes log
    CHANGES_LOG.write_text('\n'.join([f"{len(changed_files)} files changed:" ] + changed_files), encoding='utf-8')

    print(f"Processed {total} files. Changed {len(changed_files)} files.")
    print(f"Index written to: {TASK_INDEX}")
    print(f"Changes log: {CHANGES_LOG}")

if __name__ == '__main__':
    main()
