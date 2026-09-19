-- Replaces notebook Cell 13 (drop redundant/near-zero-correlation
-- columns: per_y, ad, n, om, w, ma, rms). Full cleaned + capped
-- population (~932K rows) — the default modeling set per project
-- decision (the original's 100K cap was a Colab RAM constraint; see
-- fct_asteroids_sample_100k for the legacy-parity path).

select
    spkid,
    is_pha,
    is_neo,
    class,
    class_code,
    H,
    e,
    a,
    q,
    i,
    per,
    moid
from {{ ref('int_asteroids_capped') }}
