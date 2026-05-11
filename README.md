# longest-match-first_ocr_corrector
OCR/HTR Corrector - умеет склейку, править цифры и знаки пунктуации внутри слов

Run example CLI optins:

time claude-opus-4-7-search_longest-match-first_ocr_corrector_v2_6 -i "./tr0-9.awk300.clean.window300--min-valid297.16282=2.5%.txt" -v --dict ~/prereform_words_no-bugs.1503466.txt --checkpoint-every 3000 --output window300--min-valid297.16282.max-edit-dist2.max-ngram8.jsonl --batch-size 100 --max-edit-dist 2  --max-ngram 8 --resume

```
Usage: claude-opus-4-7-search_longest-match-first_ocr_corrector_v2_6 [OPTIONS] --dict <DICT>

Options:
  -i, --input <INPUT>                                  
  -d, --dict <DICT>                                    
  -o, --output <OUTPUT>                                [default: corrected.jsonl]
  -v, --verbose                                        
      --analyze-only                                   
      --resume                                         
      --checkpoint <CHECKPOINT>                        [default: checkpoint.json.zst]
      --checkpoint-every <CHECKPOINT_EVERY>            [default: 3000]
      --fix-proper-nouns                               
      --only-proper-nouns                              
      --names-dict <NAMES_DICT>                        
      --min-word-len <MIN_WORD_LEN>                    [default: 5]
      --max-edit-dist <MAX_EDIT_DIST>                  [default: 2]
      --min-rule-freq <MIN_RULE_FREQ>                  [default: 3]
      --min-confusion-freq <MIN_CONFUSION_FREQ>        [default: 8]
      --min-ngram-freq <MIN_NGRAM_FREQ>                [default: 3]
      --max-ngram <MAX_NGRAM>                          [default: 7]
      --batch-size <BATCH_SIZE>                        [default: 1000]
      --ngram-prune-limit <NGRAM_PRUNE_LIMIT>          [default: 25000000]
      --min-learned-pair-freq <MIN_LEARNED_PAIR_FREQ>  [default: 30]
      --no-gpu                                         
  -h, --help                                           Print help
  -V, --version
  ```

  prune to save RAM - if low RAM is an issue, now common )
