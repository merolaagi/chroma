PY := $(shell command -v python 2>/dev/null || echo python3)

.PHONY: test docs results ship dry clean e5 smoke

test:            ## run the proposition suite
	PYTHONPATH=. python tests/test_propositions.py

docs:            ## regenerate figures from results/ and rebuild the PDF
	python docs/make_figures.py
	python docs/build_pdf.py

smoke:           ## 2000-step validation BEFORE committing to `make results`
	$(PY) run_e1_chunk.py chroma 2000 0
	@echo
	@echo "Check above: no '!!' lines, err settling near 0.01-0.1, dis > 0.01."
	@echo "If either alarm fired, do NOT run 'make results' -- fix first."

e5:              ## hypercube sample-efficiency comparison
	$(PY) experiments/e5_hypercube.py hotspot
	$(PY) experiments/e5_hypercube.py nk

results:         ## full E1 sweep: 5 seeds x 3 arms (overnight)
	@for s in 0 1 2 3 4; do \
	  for a in chroma ablate_hysteresis ablate_grn; do \
	    python run_e1_chunk.py $$a 20000 $$s; \
	  done; \
	done
	python aggregate.py

ship:            ## verify, rebuild docs, commit, push   (make ship M="message")
	scripts/ship.sh "$(or $(M),iterate)"

dry:             ## same as ship but stops before pushing
	scripts/ship.sh -n "$(or $(M),iterate)"

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
