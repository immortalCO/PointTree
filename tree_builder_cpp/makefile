release: builder.cpp alglibinternal.cpp alglibmisc.cpp ap.cpp dataanalysis.cpp diffequations.cpp fasttransforms.cpp integration.cpp interpolation.cpp kernels_avx2.cpp kernels_fma.cpp kernels_sse2.cpp linalg.cpp optimization.cpp solvers.cpp specialfunctions.cpp statistics.cpp
	python setup.py install

linalg: alglibinternal.cpp alglibmisc.cpp ap.cpp dataanalysis.cpp diffequations.cpp fasttransforms.cpp integration.cpp interpolation.cpp kernels_avx2.cpp kernels_fma.cpp kernels_sse2.cpp linalg.cpp optimization.cpp solvers.cpp specialfunctions.cpp statistics.cpp
	g++ -c -O3 -DAE_CPU=AE_INTEL -mavx2 -mfma alglibinternal.cpp alglibmisc.cpp ap.cpp dataanalysis.cpp diffequations.cpp fasttransforms.cpp integration.cpp interpolation.cpp kernels_avx2.cpp kernels_fma.cpp kernels_sse2.cpp linalg.cpp optimization.cpp solvers.cpp specialfunctions.cpp statistics.cpp

debug: builder.cpp alglibinternal.o alglibmisc.o ap.o dataanalysis.o diffequations.o fasttransforms.o integration.o interpolation.o kernels_avx2.o kernels_fma.o kernels_sse2.o linalg.o optimization.o solvers.o specialfunctions.o statistics.o
	g++ -O3 -DAE_CPU=AE_INTEL -mavx2 -mfma -DNOBIND -DDEBUG builder.cpp alglibinternal.o alglibmisc.o ap.o dataanalysis.o diffequations.o fasttransforms.o integration.o interpolation.o kernels_avx2.o kernels_fma.o kernels_sse2.o linalg.o optimization.o solvers.o specialfunctions.o statistics.o -o builder

nobind: builder.cpp alglibinternal.o alglibmisc.o ap.o dataanalysis.o diffequations.o fasttransforms.o integration.o interpolation.o kernels_avx2.o kernels_fma.o kernels_sse2.o linalg.o optimization.o solvers.o specialfunctions.o statistics.o
	g++ -O3 -DAE_CPU=AE_INTEL -mavx2 -mfma -DNOBIND builder.cpp alglibinternal.o alglibmisc.o ap.o dataanalysis.o diffequations.o fasttransforms.o integration.o interpolation.o kernels_avx2.o kernels_fma.o kernels_sse2.o linalg.o optimization.o solvers.o specialfunctions.o statistics.o -o builder
