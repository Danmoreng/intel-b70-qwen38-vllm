# M07: Q128 direct output

This isolated gate compares the deployed Q128 operator followed by the current
adapter's caller-output copy against the identical Q128 policy writing directly
to that caller-provided tensor. It uses the real tensor layouts, alternating
order, five warmups and 31 timed calls per arm. Serving is tested only if every
correctness shape passes and the copy removal is consistently measurable.
