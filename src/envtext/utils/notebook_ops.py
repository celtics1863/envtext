import re

def in_notebook():
    """
    检查当前环境是否是 Jupyter Notebook
    """
    try:
        # 如果能够导入 get_ipython() 函数，则说明处于 Notebook 环境中
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:
            return False
    except:
        return False
    return True