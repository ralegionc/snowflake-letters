# Data

This repository does not carry its raw inputs. The files listed below were present locally when the results in the README were produced, and were deleted on 2026-10-12 to reclaim **6.24 GB** of disk. Every one of them is a download from a public upstream source, so nothing unique was lost.

This file exists so that a re-download can be *checked* rather than assumed to match. It records the exact inventory, not just the source.

Upstream sources, with links and licences, are in the README under **Data**.

## What was removed

6.24 GB in 15 files.

| Path | Size | Files |
|---|---|---|
| `data/mascdb` | 5.72 GB | 5 |
| `data/igra` | 533.82 MB | 10 |

## Restoring it

`data/mascdb` is MASCDB v1.1 from Zenodo record 8083133; `data/igra` is IGRA2 radiosonde archives from NCEI for the Dumont d'Urville, Davis and Payerne stations. Both are fixed published releases, so a re-download should match the fingerprints below exactly. The derived arrays (`images.npy`, the simulator output and the out-of-fold parquets) were kept and are not affected.

## Inventory

`data_manifest.csv` carries every file with its exact byte length and a fingerprint. Files of 32 MiB or less have a full `sha256`. Larger files have `headtail256`: a sha256 over the first mebibyte, the last mebibyte and the exact byte length, which catches a truncated or altered download without rereading gigabytes.

| File | Size | Fingerprint |
|---|---|---|
| `data/mascdb/MASCdb.zarr.zip` | 5.38 GB | `headtail256:175e386781ffcb31388e3e573bcfa7e4239bf68368b3c48d1fb1b247cd6da5f8` |
| `data/igra/SZM00006610-data.txt` | 252.71 MB | `headtail256:821ff377111bb5afb5e5bb78b964835d5b844566979d428086a7359583e92ff9` |
| `data/igra/AYM00089571-data.txt` | 113.89 MB | `headtail256:29ea9436ddb0b6d2180f3ed38d097c0ad242c75d3e7941ef795cd56905c6aac4` |
| `data/mascdb/MASCdb_cam2.parquet` | 104.74 MB | `headtail256:b572fe746bc25a4e7062f74bc9c8b6c127556bb4e6d9a9acfa7ac076f8efeea5` |
| `data/mascdb/MASCdb_cam0.parquet` | 104.72 MB | `headtail256:a02baab620acda7ba13055c02d9c89b6f0d40d6eb74b084bb02923e5a7c7e9fd` |
| `data/mascdb/MASCdb_cam1.parquet` | 104.65 MB | `headtail256:483a1b84285a8a25331386ce7113a36fa3aca63b9da0b855fc5e26a72da01213` |
| `data/igra/SZM00006610-data.txt.zip` | 62.02 MB | `headtail256:27c8eedf54c774e68035d7797fca36f825e47421846439885355303125e41aeb` |
| `data/igra/AYM00089642-data.txt` | 38.64 MB | `headtail256:491188e2965dae4a46a7da507806aa836a1bcfcfb6279365185a33598d48350a` |
| `data/mascdb/MASCdb_triplet.parquet` | 31.29 MB | `sha256:0b78fa8084f46f7566ad426312c81fd6c015eee41a05e0b217f213ee73a2fdec` |
| `data/igra/AYM00089571-data.txt.zip` | 25.17 MB | `sha256:e597ce73f7e2eb677c1cc132cf17e7602962c10a2bff4bbe9ec23586c1bd3cca` |
| `data/igra/SZM00006610.parquet` | 13.46 MB | `sha256:d19b480f7b16c965ae07708256444f7337803ac41d6e4ffabc612263f4810989` |
| `data/igra/AYM00089571.parquet` | 11.78 MB | `sha256:92ea6171f11c4e4f9b28c891dd6040dca6cc94cc281b752431d835d5b5f3e869` |
| `data/igra/AYM00089642-data.txt.zip` | 9.40 MB | `sha256:e6bc5e8ad94b74004201d4a14959164253fb622f2aa75f18a79b935b92f7c05f` |
| `data/igra/AYM00089642.parquet` | 6.50 MB | `sha256:978b43d310790d2015bbabc137ab465f7b437ec773bb8418c8e5898b9095949e` |
| `data/igra/igra2-station-list.txt` | 254.83 KB | `sha256:a13843c639c936b7c820a6928a0d62c3f3ac34bb94912aff89088cb9815c5498` |

A source that is live rather than a fixed release will not reproduce these fingerprints. Where that applies it is said above.

