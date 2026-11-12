FROM sha256:f8b0ef94cf95cfee20b0c5500db1173fdaaca4af4e3cbaa352a511e78e6a0c6d
COPY cow_copy_fix.py /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/b70_cow_copy_fix.py
COPY install_release.py /opt/b70/cache1792/install_release.py
COPY production_policy.json production_policy.sha256 /opt/b70/cache1792/
RUN /opt/venv/bin/python /opt/b70/cache1792/install_release.py
LABEL org.local.b70.policy.sha256="997d6b63458b312c038a1bd89034487d547e0368303c648aef6c44c69e9cae9f" org.local.b70.release-status="PRODUCTION"
ENV B70_POLICY_SHA256=997d6b63458b312c038a1bd89034487d547e0368303c648aef6c44c69e9cae9f
