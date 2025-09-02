# GNU Radio package initialization
# Import system gnuradio components first
import sys
import pkgutil

# Allow importing from system gnuradio installation
__path__ = pkgutil.extend_path(__path__, __name__)
