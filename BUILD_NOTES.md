# BUILD NOTES — gta6-trophy-rarity-forecast-01, slot 1

## Entry
- File: `main.py` (261 baris, Python 3.13 stdlib saja, tanpa network, deterministik,
  ~0,05 detik/case — jauh di bawah batas 8 detik).
- Sesi build ini dimulai 18:40 WIB 19 Nov 2026, SETELAH bounty buka (18:30 WIB) —
  sah di-enter. JANGAN pakai kode dari sesi scoping (16:44 WIB).
- Belum di-submit, belum di-push, belum ada repo — parent yang buat repo (HARUS
  dibuat setelah 18:30 WIB) + LICENSE MIT (starter) + submit.

## Model (ringkas)
Median per trophy = blend 0,6/0,2/0,2 dari:
1. **Ceiling-logistic** — `r(u) = c·sigmoid(a+b·u)`, `u = ln(1+jam sejak launch)`;
   ceiling `c` di-grid (di atas rate tertinggi yang teramati s.d. 99,99), `(a,b)`
   dari weighted least squares atas `logit(r/c)`. Ini kuncinya: kurva rate
   MENJENUH ke ceiling, sementara garis lurus di logit meledak ke ~100%.
2. **Logit trend** — `logit(rate)` vs `ln(1+t)`, recency-weighted (half-life 24 jam).
3. **Persistence** — rate terakhir, sebagai anchor.
History dulu dikompres ke distinct page states (refresh Exophase tiap beberapa
jam → ratusan read = puluhan plateau). Spread = `0,85 × 0,5·√hari` + guard
stale-page + lag refresh, clamp [0,4–2,2] logit.
**Stale-page guard**: kalau players-tracked tidak bergerak sama sekali dalam
history (page basi — persis kondisi practice), median digeser ke bawah (dilusi:
pemain baru datang hampir tanpa trophy) dan spread dilebarkan. Kalau history
bergerak normal, guard tidak bekerja.

## Validasi lokal (bukan grade resmi — scorer tetap di server)
- Practice p00/p01 vs jawaban REKONSTRUKSI (dari total refresh yang dipublikasi:
  3.868 pemain / 4.276 trophy; trophy 100% tetap, sisanya ∝ earned): loss
  **0,215 / 0,202** vs house baseline **1,401**, coverage 0,98. (Jawaban
  rekonstruksi itu aproksimasi, bukan jawaban resmi — tapi selisihnya besar.)
- Backtest final sintetis (7 skenario × 3 seed: progres cepat/lambat, P tumbuh
  cepat/lambat, refresh jarang/sering; dunia simulasi: P log-logistik, tiap
  trophy delay lognormal + ceiling, refresh Poisson): mean loss **0,402** vs
  persistence **0,750** dan curve baseline **0,707**. Skenario terburuk 0,670
  (refresh jarang).
- Format: valid (tepat 1 key `trophies`, 55 baris, q non-decreasing, ≤64 KiB);
  deterministik (byte-identik di re-run); lolos `python3 -P -s`; 8 edge case
  lolos (1 read, rate null/0%, horizon 10 mnt–30 hari, trophy ke-56 muncul
  telat, players_tracked bolong).

## Yang TIDAK diuji / risiko
- **House AI model** (superbot, dibuat 21 Nov 09:30 UTC dengan data s.d. saat
  itu) adalah bar sesungguhnya — tidak bisa diuji lokal; model ini harus
  mengalahkannya juga, bukan cuma dua baseline. Backtest sintetis menunjukkan
  keunggulan besar atas baseline, tapi dunia nyata (rezim main weekend vs
  weekday, kecepatan refresh Exophase, panjang cerita GTA VI) bisa berbeda
  dari simulasi saya.
- Kalau di final history ternyata datar/mandek (recorder bermasalah), guard
  stale-page akan aktif dan median turun — itu perilaku yang diinginkan.
- Jangan submit dari angka practice: practice 90 menit "little guide to the
  final's week" (kata RULES sendiri).

## Sebelum submit (parent)
1. Buat repo publik baru (setelah 18:30 WIB), isi `main.py` + LICENSE MIT.
2. Kirim isi entry ke user dulu (txt/md) — aturan review-sebelum-submit.
   Setelah user gas/OK, bundle transcript sesi INI + submit.
3. File lokal (bukan bagian entry): `local/score.py`, `local/simulate.py`,
   `local/candidates.py`, `local/tune_spread.py`, `local/tune_final.py`,
   `local/test_entry.py`, `local/test_edge.py`, `local/diagnose.py`,
   `out/p00.out`, `out/p01.out`.
