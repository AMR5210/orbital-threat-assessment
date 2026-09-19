-- Replaces the runtime-only cap thresholds computed in notebook Cell 9
-- and re-derived (redundantly) again in Cell 33. Those thresholds were
-- printed to stdout and never saved anywhere durable — this materializes
-- them as a queryable table instead.
--
-- Per project decision: caps are computed from the full cleaned
-- population (this model, ~932K rows), not the 100K sample the
-- original notebook capped against. fct_asteroids_sample_100k joins
-- against this same table, so the sample and full-population paths
-- share one definition of "99th percentile" rather than each computing
-- their own (as Cells 9 and 33 did).

-- percentile_cont (exact, linear-interpolated — matches pandas
-- .quantile() default) rather than approx_percentile: at ~932K rows
-- there's no performance reason to approximate a threshold other
-- stages depend on. Uses the ANSI ordered-set-aggregate syntax
-- (PERCENTILE_CONT(frac) WITHIN GROUP (ORDER BY col)) rather than
-- DuckDB's quantile_cont(col, frac) shorthand, since DuckDB supports
-- both but Snowflake only recognizes the ANSI form -- one statement
-- that works unmodified against both targets (found via a real
-- `dbt build -t prod` failure: "Unknown functions QUANTILE_CONT").
select
    percentile_cont(0.99) within group (order by a)    as a_p99,
    percentile_cont(0.99) within group (order by q)    as q_p99,
    percentile_cont(0.99) within group (order by moid) as moid_p99,
    percentile_cont(0.99) within group (order by i)    as i_p99
from {{ ref('stg_asteroids') }}
