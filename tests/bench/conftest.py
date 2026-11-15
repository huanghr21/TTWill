"""让官方依赖只读随包价目表，防止导入时请求外部服务。"""

import os

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
