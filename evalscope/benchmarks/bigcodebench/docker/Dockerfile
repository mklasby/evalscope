# BigCodeBench's evaluation image (v0.2.4, by digest) plus the NLTK data its tasks load, which the image lacks (upstream
# downloads it at run time; our sandboxes have no network): the packages in nltk_data.lock, at a pinned nltk_data
# commit, each checked against its SHA-256, installed into /usr/local/share/nltk_data, on NLTK's search path.
# The base image's entrypoint runs BigCodeBench's evaluator; the sandbox clears it.
FROM bigcodebench/bigcodebench-evaluate@sha256:a3cd34ec3840a49d6b7afb240f4bdd47c350bc5991043fd0a91773830f7cd405

USER root
COPY fetch_nltk_data.py nltk_data.lock /tmp/nltk_fetch/
RUN python3 /tmp/nltk_fetch/fetch_nltk_data.py /tmp/nltk_fetch/nltk_data.lock /usr/local/share/nltk_data \
    && rm -rf /tmp/nltk_fetch
USER bigcodebenchuser
