from symbolfit.symbolfit import *
from symbolfit.Dataset import Dataset
import pandas as pd

indir       = "/home/shared/B02KstMuMu/Run3/b02kstmumu-run3/output"
filename    = "mB_vs_q2_constrained.root"

# FIXME: crashes if set to True. Cannot currently load previous results and need to rerun
# the whole procedure everytime.
load_results = False

dataset = Dataset()
dataset.create_dataset_from_TH2D(indir, filename, cuts=[[(3.24, 11.56),], [(5440, 5800),]])
# dataset.create_dataset_from_TH2D(indir, filename, blind=[[(0, 20), (5200, 5100)], [(7, 10), (5100, 5200)]])

pysr_config = importlib.import_module('examples.pysr_configs.pysr_config_gauss').pysr_config

output_dir = "output_dir/"

model = SymbolFit(
    x = dataset.x,
    y = dataset.y,
    y_up = dataset.y_up,
    y_down = dataset.y_down,
    pysr_config = pysr_config,
    max_complexity = 60,
    input_rescale = True,
    scale_y_by = 'mean',
    max_stderr = 20,
    fit_y_unc = True,
    random_seed = None,
    loss_weights = None
)
 
if load_results:

    dtypes = {'Complexity' :                         np.int64,
            'PySR equation' :                        object,
            'Parameterized equation' :               object,
            'Parameterization' :                     object,
            'Parameters: (best-fit, +1, -1)' :       object,
            'Covariance' :                           object,
            'Correlation' :                          object,
            'Parameterized equation, unscaled':      object,
            'RMSE':                                  np.float64,
            'R2' :                                   np.float64,
            'RMSE (before ROF)' :                    np.float64}

    model.func_candidates = pd.read_csv(f"{output_dir}/candidates.csv", index_col=0, dtype=dtypes)

else:
    model.fit()

    model.save_to_csv(output_dir = output_dir)


model.plot_to_pdf(
    output_dir = 'output_dir_train/',
    # bin_widths_1d = dataset.bin_widths_1d,
    plot_logy = False,
    # plot_logx = False,
    sampling_95quantile = False,
    bin_edges_2d = dataset.bin_edges_2d,
    plot_logx0 = False,
    plot_logx1 = False,
    #cbar_min = None,
    #cbar_max = None,
    #cmap = None,
    #contour = None,
    # ^ additional options for 2D plotting
)



# Plot with full dataset
dataset_full = Dataset()
# dataset_full.create_dataset_from_TH2D(indir, filename, cuts=[(3.24, 11.56), (5100, 5800)])
dataset_full.create_dataset_from_TH2D(indir, filename, blind=[[(0, 20), (5200, 5100)], [(7, 10), (5100, 5200)]])

model.x         = dataset_full.x
model.y         = dataset_full.y
model.y_up      = dataset_full.y_up
model.y_down    = dataset_full.y_down

model.plot_to_pdf(
    output_dir = 'output_dir_test/',
    # bin_widths_1d = dataset.bin_widths_1d,
    plot_logy = True,
    # plot_logx = False,
    sampling_95quantile = False,
    bin_edges_2d = dataset_full.bin_edges_2d,
    plot_logx0 = False,
    plot_logx1 = False,
    #cbar_min = None,
    #cbar_max = None,
    #cmap = None,
    #contour = None,
    # ^ additional options for 2D plotting
)
