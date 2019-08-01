nn.so: *.f90
	gfortran -ffree-form -fPIC -O3 -c NN_data_structures.f90
	gfortran -ffree-form -fPIC -O3 -c NN_activation_functions.f90
	gfortran -ffree-form -fPIC -O3 -c NN_loss_functions.f90
	gfortran -ffree-form -fPIC -O3 -c NN_optimization_algorithms.f90 
	gfortran -ffree-form -fPIC -O3 -c NN_propagate.f90 
	gfortran -ffree-form -fPIC -O3 -c NN_sens_predict.f90 
	gfortran -ffree-form -fPIC -O3 -c NN_train.f90 
	gfortran -ffree-form -fPIC -O3 -c NN_python_wrapper.f90
	f2py -m nn -h nn.pyf NN_python_wrapper.f90 --overwrite-signature
	f2py -c nn.pyf *.o
	rm  *.o
	rm nn.pyf
	rm *.mod
