# Tab layout builders — each returns a dict of Gradio components for event wiring.
from ui.tabs.data_tab       import build_data_tab        # noqa: F401
from ui.tabs.train_tab      import build_train_tab       # noqa: F401
from ui.tabs.inference_tab  import build_inference_tab   # noqa: F401
from ui.tabs.rlhf_tab       import build_rlhf_tab        # noqa: F401
from ui.tabs.evaluation_tab import build_evaluation_tab  # noqa: F401
from ui.tabs.gguf_tab       import build_gguf_tab        # noqa: F401
from ui.tabs.share_tab      import build_share_tab       # noqa: F401
