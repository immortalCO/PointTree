from setuptools import setup
from torch.utils import cpp_extension as cpp
import datetime

ind = 0

def get_version():
    return f'{datetime.date.today().strftime("%Y.%m.%d")}.{ind}'

while True:
    ind += 1
    try:
        file = open("versions/" + get_version() + ".placeholder")
        file.close()
        continue
    except:
        break

with open("versions/" + get_version() + ".placeholder", 'w') as file:
    file.write(get_version())


setup(
    name='tree_builder_cpp',
    version=get_version(),
    ext_modules=[
        cpp.CppExtension('tree_builder_cpp', 
            ["builder.cpp", 
            "alglibinternal.cpp", "alglibmisc.cpp", "ap.cpp", "dataanalysis.cpp", 
            "diffequations.cpp", "fasttransforms.cpp", "integration.cpp", "interpolation.cpp", 
            "kernels_avx2.cpp", "kernels_fma.cpp", "kernels_sse2.cpp", "linalg.cpp", 
            "optimization.cpp", "solvers.cpp", "specialfunctions.cpp", "statistics.cpp", ],
            extra_compile_args={"cxx": ["-O3"]})
    ],
    cmdclass={"build_ext": cpp.BuildExtension}
)