# M06: 196K preemption removal

One-variable candidate: production Q128/4096/MTP4/180 W plus
`--prefix-cache-retention-interval 0`. The frozen M01 196K run is the control.

The background runner first requires fewer preemptions, less scheduler replay
and at least 3% lower TTFT. Only a passing screen gets a second 196K run and a
fresh-engine cold/warm shared-prefix state-correctness test. Production is
restored after every outcome; promotion is a separate reviewed action.
