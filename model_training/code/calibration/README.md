# Calibration and measurement scripts

Every tuned constant in the racing code cites, in its comment, the script
that measured it. Those scripts are here. Run each from this folder under
the memory cap, for example `python ../memcap.py 3000 voronoi_rule.py`.
Each prints the table quoted in the comment. On Linux or macOS, run
`python voronoi_rule.py` directly instead (`memcap.py` uses a Windows Job
Object).

| Script | Feeds | Where the numbers are quoted |
|---|---|---|
| validate.py | before/after check of any car or quality change | thesis/HISTORY.md, part "Thesis status", CODE_EXPLAINED.md |
| control_targets.py | control targets, domain ends, random hit counts | racing/problem.py, `_CONTROL_TARGETS` |
| tile_weights.py | tile WFC weights against the current quality function | `_WFC_WEIGHTS` / `_weights` in the four tile modules |
| voronoi_rule.py | Voronoi weight ceiling and site count | racingvoronoi/problem.py, `_WEIGHT_MAX_M2` |
| corner_census.py, census_dump.py | the corner census of the 24 circuits (census_dump writes census_corners.json) | racingtile/problem.py vocabulary table |
| vocab_share.py, mix_sweep.py, tile_catalog_fn.py, corner_classes.py | tile vocabulary against the census; built corner mix | the four tile modules |
| proto_sq.py, proto_hx.py, fit_check.py, chicane_census.py | tile shape ranges, hex variants, road clearance, the hex S chicane | racingtile, racingtilehex, racingtilediag, racingtilehexdiag |
| tile_sweep.py, qcand4.py | square-tile transition length | racingtile/problem.py, `_TRANSITION_M` |
| porsche_fit.py | drag area and driveline efficiency | racing/engine.py |
| nhtsa_cgroof.py (reads nhtsa1999.pdf) | cg height ratio | racing/engine.py, `h_cg` |
| oversteer_cg.py | power-on body slip against cg height | racing/engine.py, `h_cg` |
| driver_sweep.py, driver_table.py | path-follower settings; default driver comparison | racing/agent.py; racing/problem.py, driver comment |
| specificity.py | the quality function on tracks built to be implausible | racing/problem.py, `_TYPICALITY` comment; CODE_EXPLAINED.md 3.5 |
| robustness.py | the verdict and the ranking under other reasonable quality settings | the same |
| torcs_circuits.py | the reference circuits raced in TORCS: the clean-race rule of the TORCS driver | racing/problem.py, `_torcs_summary` |
| qctl_cand.py | candidate control screen | racing/problem.py, controls kept out (needs saved data that is not in the repository; see its docstring) |

`../calibrate_typicality.py` prints the `_TYPICALITY` constants. `tests/test_racing.py`
fails if the constants in the code differ from what it fits.

nhtsa1999.pdf is not in the repository: it is Heydinger et al. (1999),
*Measured Vehicle Inertial Parameters: NHTSA's Data Through November 1998*,
SAE 1999-01-1336 (doi:10.4271/1999-01-1336), whose copyright is SAE's. To
rerun nhtsa_cgroof.py, obtain the paper and save it here under that name.
