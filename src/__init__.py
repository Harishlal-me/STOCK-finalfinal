"""
Stock Prediction Model Package
"""

__version__ = "1.0.0"
__author__ = "Stock Prediction Team"
__description__ = "Multi-Task LSTM Stock Prediction Model"

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == 'data_loader':
        from src import data_loader
        return data_loader
    elif name == 'feature_engineer':
        from src import feature_engineer
        return feature_engineer
    elif name == 'model_builder':
        from src import model_builder
        return model_builder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'data_loader',
    'feature_engineer', 
    'model_builder',
    'get_version',
    'get_info'
]


def get_version():
    """Get package version"""
    return __version__


def get_info():
    """Get package information"""
    return {
        'version': __version__,
        'author': __author__,
        'description': __description__
    }