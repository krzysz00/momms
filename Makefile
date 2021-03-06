.PHONY: build results plots only-plots
EXPERIMENT=default

SRC = $(wildcard src/*.rs) build.rs
VARIANTS=gemm3 gemm3_ab_bc_kernel gemm3_rectangles gemm3_parallel

_CARGO_OUT = target/release/exper_
DATADIR=results/$(EXPERIMENT)

all: plots

build: $(SRC)
	cargo build --release --features "blis hsw asm_snippets"

$(addprefix $(CARGO_OUT),$(VARIANTS)): build

$(DATADIR):
	mkdir $@

$(DATADIR)/%.dat: $(_CARGO_OUT)% | $(DATADIR)
	@echo $<
	./$< | tee $@
	./results/percent_change_script.sh < $@ | tee $(DATADIR)/$*_changes.dat

results: $(addprefix $(DATADIR)/,$(addsuffix .dat,$(VARIANTS)))

PLOTEXT=pdf
only-plots: $(wildcard results/plot_*.py)
	./results/plot_experiment.py $(DATADIR)/gemm3.dat "gemm3()" "\$$G\$$ += \$$D(EF)\$$, square matrices" "N" $(DATADIR)/gemm3.$(PLOTEXT)
	./results/plot_experiment.py $(DATADIR)/gemm3_ab_bc_kernel.dat "gemm3()" "\$$G^T\$$ += \$$F^T(E^TD^T)\$$, square matrices" "N" $(DATADIR)/gemm3_ab_bc_kernel.$(PLOTEXT)
	./results/plot_experiment.py $(DATADIR)/gemm3_parallel.dat "gemm3()" "\$$G\$$ += \$$D(EF)\$$, square matrices, 4 cores" "N" $(DATADIR)/gemm3_parallel.$(PLOTEXT)
	./results/plot_multi_shape_experiment.py $(DATADIR)/gemm3_rectangles.dat "gemm3()" "\$$G\$$ += \$$D(EF)\$$, narrow dimension = 252" $(DATADIR)/gemm3_rectangles.$(PLOTEXT)

	./results/plot_experiment_memory.py $(DATADIR)/gemm3.dat "gemm3()" "Memory usage of gemm3() vs. BLIS algorithm, square matrices" "N" $(DATADIR)/gemm3_memory.$(PLOTEXT)
	./results/plot_multi_shape_experiment_memory.py $(DATADIR)/gemm3_rectangles.dat "gemm3()" "Memeroy usage of right paranthesized kernel vs. BLIS algo., narrow dimension = 252" $(DATADIR)/gemm3_rectangles_memory.$(PLOTEXT)

	./results/plot_percent_change.py $(DATADIR)/gemm3_changes.dat "\$$G\$$ += \$$D(EF)\$$, square matrices" "N" $(DATADIR)/gemm3_changes.$(PLOTEXT)
	./results/plot_percent_change.py $(DATADIR)/gemm3_ab_bc_kernel_changes.dat "\$$G^T\$$ += \$$F^T(E^TD^T)\$$, square matrices" "N" $(DATADIR)/gemm3_ab_bc_kernel_changes.$(PLOTEXT)
	./results/plot_percent_change.py $(DATADIR)/gemm3_parallel_changes.dat "\$$G\$$ += \$$D(EF)\$$, square matrices, 4 cores" "N" $(DATADIR)/gemm3_parallel_changes.$(PLOTEXT)

plots: results only-plots
