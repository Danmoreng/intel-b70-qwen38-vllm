# E04a reduction port

Ported only the reduction scheduler/grid and disjoint head-dimension output
loop onto clean M04 header pin 6d92b1b..., preserving the original exp2,
unnormalized partial-output convention. Original C++ wrapper, masks, scratch,
q2–5 contract, split counts and tile8 are unchanged. Compile/import passed.

The candidate passes existing attention causal and graph tests plus 40
adversarial whole-call cases: B1/B4, R2–5, histories 4097/32769/65537/131073/
261121, unequal siblings, inactive splits, changing Q, device lengths and page
tables across graph replay. All 40 pre-mutation outputs are bitwise identical
to the original operator; mutated replay agrees with native references.
Short histories also use an independent FP32 oracle and original .01/.003
tolerances.

One exploratory whole-call graph timing block shows practically no long-q4
benefit (C1 64K around +0.09%, C4 long cells around zero to slight loss).
These are exploratory numbers, not confidence-qualified gains. The small
short-q4 reduction saving cannot produce a material whole-model improvement.
Stop this candidate before a full serving matrix: NO_MATERIAL_GAIN. Keep the
patch and receipts available, and retain original M04 in production.
