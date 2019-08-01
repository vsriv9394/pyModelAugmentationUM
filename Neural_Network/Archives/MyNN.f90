module NN

    implicit none
    contains

!::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
!
!   Neural Network module interfaced with python for FIML
!
!::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    
    
    
    
    
    
    subroutine activate(act_fn, value)

        implicit none

    !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    !
    !   Derivative for activation function for the Neural Network
    !
    !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

        
        CHARACTER(LEN=25), INTENT(IN)    :: act_fn
        REAL*8,            INTENT(INOUT) :: value

        if ( act_fn == 'sigmoid' ) then

            call sigmoid(value)
        
        elseif ( act_fn == 'relu' ) then
        
            call relu(value)
        
        else
        
            write(*,*) "Unknown activation function name provided. Valid options are"
            write(*,*) "- relu"
            write(*,*) "- sigmoid"
            stop
        
        end if
        
    end subroutine activate







    function d_act(act_fn, value)

        implicit none

    !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    !
    !   Derivative for activation function for the Neural Network
    !
    !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        
    

    !   Dummy argument

        REAL*8 :: d_act


    
    !   Argument initialization

        CHARACTER(LEN=25) :: act_fn
        REAL*8 :: value

    
    
    !   Function definition

        if ( act_fn == 'sigmoid' ) then

            d_act = d_sigmoid(value)
        
        elseif ( act_fn == 'relu' ) then
        
            d_act =  d_relu(value)
        
        else
        
            write(*,*) "Unknown activation function name provided. Valid options are"
            write(*,*) "- relu"
            write(*,*) "- sigmoid"
            stop
        
        end if
        
    
    
    end function d_act






    
    subroutine relu(value)

        implicit none

    !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    !
    !   Activation function for the Neural Network - Rectified Linear Unit
    !
    !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        
    
    
    !   Argument initialization

        REAL*8, INTENT(INOUT) :: value

    
    
    !   Function definition

        if ( value<0.0 ) then
            value = 0.0
        end if
        
    
    
    end subroutine relu







    subroutine sigmoid(value)

        implicit none

    !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    !
    !   Activation function for the Neural Network - Sigmoid
    !
    !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        
    
    
    !   Argument initialization

        REAL*8, INTENT(INOUT) :: value

    
    
    !   Function definition

        value = 1.0 / (1.0 + EXP(-value))
        
    
    
    end subroutine sigmoid

    
    
    
    
    
    
    function d_relu(value)

        implicit none

    !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    !
    !   Derivative for activation function for the Neural Network - Rectified Linear Unit
    !
    !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        
    

    !   Dummy argument

        REAL*8 :: d_relu


    
    !   Argument initialization

        REAL*8 :: value

    
    
    !   Function definition

        if ( value>0.0 ) then
            d_relu = 1.0
        else
            d_relu = 0.0
        end if
        
    
    
    end function d_relu

    
    
    
    
    
    
    function d_sigmoid(value)

        implicit none

    !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    !
    !   Derivative for activation function for the Neural Network - Sigmoid
    !
    !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        
    
    
    !   Dummy argument

        REAL*8 :: d_sigmoid


    
    !   Argument initialization

        REAL*8 :: value

    
    
    !   Function definition

        d_sigmoid = value * (1.0 - value)
        
    
    
    end function d_sigmoid

    
    
    
    
    
    
    subroutine forwprop(act_fn, n_layers, n_neurons_layer, n_features, n_weights_network, weights, n_neurons_network, neurons)

        implicit none

    !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    !
    !   Prediction (forward propagation) function for the neural network, given features
    !
    !   weights = {bias, weight_1, ..., weight_n} for all neurons in a layer for all layers in the network
    !   neurons = activated value for all neurons in a layer for all layers in the network
    !
    !::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    
    
    !   Argument initialization

        CHARACTER(LEN=25), INTENT(IN)    :: act_fn
        INTEGER,           INTENT(IN)    :: n_layers, n_features, n_weights_network, n_neurons_network
        INTEGER,           INTENT(IN)    :: n_neurons_layer(n_layers)

        REAL*8,            INTENT(IN)    :: weights(n_weights_network)
        REAL*8,            INTENT(INOUT) :: neurons(n_neurons_network)

    
    !------------------------------------------------------------------------------------------------------------

    
    !   Local variables initialization

        INTEGER :: i_layer, i_neuron
        INTEGER :: n_neurons_prev_layer, weight_counter, neuron_counter, prev_layer_beg

        !--- Initialize the number of neurons in the previous layer to number of features ---!

        n_neurons_prev_layer = n_features
        

        !--- Initialize the parameter and output counter to 1 and last layer index to 0---!

        weight_counter   = 1
        neuron_counter   = n_features+1
        prev_layer_beg   = 0

    
    !------------------------------------------------------------------------------------------------------------

    
    !   Function definition

        !--- Loop over layers ---!

        do i_layer = 1, n_layers
     
            
            !--- Loop over the neurons in this layer to evaluate outputs for every one of these ---!
            
            do i_neuron = 1, n_neurons_layer(i_layer)


                !--- Initialize the outputs for this neuron for every data point to bias ---!

                neurons(neuron_counter) = weights( weight_counter ) + &
                                          SUM( weights( weight_counter+1 : weight_counter+n_neurons_prev_layer ) * &
                                               neurons( prev_layer_beg+1 : prev_layer_beg+n_neurons_prev_layer ))
                
                call activate(act_fn, neurons(neuron_counter))
                
                weight_counter = weight_counter + 1 + n_neurons_prev_layer

                neuron_counter = neuron_counter + 1


            end do

            n_neurons_prev_layer = n_neurons_layer(i_layer)
            prev_layer_beg       = neuron_counter - n_neurons_prev_layer

        end do

        neurons(neuron_counter) = weights( weight_counter ) + &
                                  SUM( weights( weight_counter+1 : weight_counter+n_neurons_prev_layer ) * &
                                       neurons( prev_layer_beg+1 : prev_layer_beg+n_neurons_prev_layer ))

    
    
    end subroutine forwprop







    subroutine backprop(act_fn, n_layers, n_neurons_layer, n_features, n_weights_network, weights, d_weights, &
                                                                       n_neurons_network, neurons, d_neurons)

        implicit none

    !:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    !
    !   Prediction (forward propagation) function for the neural network, given features
    !
    !:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    
    !   Argument initialization

        CHARACTER(LEN=25), INTENT(IN)    :: act_fn
        INTEGER,           INTENT(IN)    :: n_layers, n_features, n_weights_network, n_neurons_network
        INTEGER,           INTENT(IN)    :: n_neurons_layer(n_layers)

        REAL*8, INTENT(IN)    :: weights(n_weights_network)
        REAL*8, INTENT(IN)    :: neurons(n_neurons_network)
        REAL*8, INTENT(INOUT) :: d_weights(n_weights_network)
        REAL*8, INTENT(INOUT) :: d_neurons(n_neurons_network)

    
    !-----------------------------------------------------------------------------------------------------------------------------
    
    !   Local variables initialization

        INTEGER :: i_layer, i_neuron
        INTEGER :: weight_counter, neuron_counter, prev_layer_beg, n_weights_neuron
        

        !--- Initialize the parameter and output counter to 1 and last layer index to 0---!

        n_weights_neuron = n_neurons_layer(n_layers)                    ! Change the # of weights to # of neurons previous layer
        weight_counter   = n_weights_network - n_weights_neuron         ! Set weight counter to the bias value
        neuron_counter   = n_neurons_network                            ! Set neuron counter to the current neuron
        prev_layer_beg   = n_neurons_network - n_weights_neuron - 1     ! Set to neuron before previous layer's first neuron 

    
    !-----------------------------------------------------------------------------------------------------------------------------

    !   Function definition

        d_neurons(:neuron_counter-1) = 0.0

        d_weights(weight_counter) = d_neurons(neuron_counter)
        d_weights(weight_counter+1:weight_counter+n_weights_neuron) = d_weights(weight_counter) * &
                                                                      neurons(prev_layer_beg + 1 : &
                                                                              prev_layer_beg + n_weights_neuron)
        
        d_neurons(prev_layer_beg+1:prev_layer_beg+n_weights_neuron) = &
        d_neurons(prev_layer_beg+1:prev_layer_beg+n_weights_neuron) + d_weights(weight_counter) * &
                                                                      weights(weight_counter + 1 : &
                                                                              weight_counter + n_weights_neuron)


        do i_layer = n_layers, 2, -1

            n_weights_neuron = n_neurons_layer(i_layer-1)
            weight_counter   = weight_counter - (n_weights_neuron + 1) * n_neurons_layer(i_layer)
            prev_layer_beg   = prev_layer_beg -  n_weights_neuron
            neuron_counter   = neuron_counter -  n_neurons_layer(i_layer)

            do i_neuron = 1, n_neurons_layer(n_layers)

                d_weights(weight_counter) = d_act(act_fn, neurons(neuron_counter)) * d_neurons(neuron_counter)
                
                d_weights(weight_counter+1:weight_counter+n_weights_neuron) = d_weights(weight_counter) * &
                                                                              neurons(prev_layer_beg + 1 : &
                                                                                      prev_layer_beg + n_weights_neuron)

                d_neurons(prev_layer_beg+1:prev_layer_beg+n_weights_neuron) = &
                d_neurons(prev_layer_beg+1:prev_layer_beg+n_weights_neuron) + d_weights(weight_counter) * &
                                                                              weights(weight_counter + 1 : &
                                                                                      weight_counter + n_weights_neuron)
                
                neuron_counter = neuron_counter + 1
                weight_counter = weight_counter + 1 + n_neurons_layer(i_layer-1)

            end do

            weight_counter = weight_counter - (n_weights_neuron + 1) * n_neurons_layer(i_layer)
            neuron_counter = neuron_counter - n_neurons_layer(i_layer)

        end do

        n_weights_neuron = n_features
        weight_counter   = 1
        prev_layer_beg   = 0
        neuron_counter   = n_features+1

        do i_neuron = 1, n_neurons_layer(1)

            d_weights(weight_counter) = d_act(act_fn, neurons(neuron_counter)) * d_neurons(neuron_counter)
            
            d_weights(weight_counter+1:weight_counter+n_weights_neuron) = d_weights(weight_counter) * &
                                                                          neurons(prev_layer_beg + 1 : &
                                                                                  prev_layer_beg + n_weights_neuron)

            d_neurons(prev_layer_beg+1:prev_layer_beg+n_weights_neuron) = &
            d_neurons(prev_layer_beg+1:prev_layer_beg+n_weights_neuron) + d_weights(weight_counter) * &
                                                                          weights(weight_counter + 1 : &
                                                                                  weight_counter + n_weights_neuron)
            
            neuron_counter = neuron_counter + 1
            weight_counter = weight_counter + 1 + n_features
            

        end do
        
        
    end subroutine backprop







    subroutine predict(act_fn, n_layers, n_neurons_layer, n_features, n_weights_network, weights, n_neurons_network, neurons, &
                                                                                                        n_data, features, beta)

        implicit none

        CHARACTER(LEN=25), INTENT(IN)    :: act_fn
        INTEGER,           INTENT(IN)    :: n_layers, n_features, n_data, n_weights_network, n_neurons_network
        INTEGER,           INTENT(IN)    :: n_neurons_layer(n_layers)

        REAL*8, INTENT(IN)    :: weights(n_weights_network)
        REAL*8, INTENT(INOUT) :: neurons(n_neurons_network)
        REAL*8, INTENT(IN)    :: features(n_features, n_data)
        REAL*8, INTENT(OUT)   :: beta(n_data)

        INTEGER :: i_data

        do i_data = 1, n_data

            neurons(1:n_features) = features(:, i_data)
            call forwprop(act_fn, n_layers, n_neurons_layer, n_features, n_weights_network, weights, n_neurons_network, neurons)
            beta(i_data) = neurons(n_neurons_network)

        end do

    end subroutine predict







    subroutine get_sens(act_fn, n_layers, n_neurons_layer, n_features, n_weights_network, weights, n_neurons_network, neurons, &
                                                                                            n_data, features, d_beta, d_weights)

        implicit none

        CHARACTER(LEN=25), INTENT(IN)    :: act_fn
        INTEGER,           INTENT(IN)    :: n_layers, n_features, n_data, n_weights_network, n_neurons_network
        INTEGER,           INTENT(IN)    :: n_neurons_layer(n_layers)

        REAL*8, INTENT(IN)    :: weights(n_weights_network)
        REAL*8, INTENT(INOUT) :: neurons(n_neurons_network)
        REAL*8, INTENT(IN)    :: features(n_features, n_data)
        REAL*8, INTENT(IN)    :: d_beta(n_data)
        REAL*8, INTENT(OUT)   :: d_weights(n_weights_network)

        INTEGER :: i_data
        REAL*8  :: d_neurons(n_neurons_network)
        REAL*8  :: d_weights_dummy(n_weights_network)

        d_weights(:) = 0.0

        do i_data = 1, n_data

            neurons(1:n_features)        = features(:, i_data)
            d_neurons(n_neurons_network) = d_beta(i_data)
            call forwprop(act_fn, n_layers, n_neurons_layer, n_features, n_weights_network, weights, n_neurons_network, neurons)
            call backprop(act_fn, n_layers, n_neurons_layer, n_features, n_weights_network, weights, d_weights_dummy, &
                                                                         n_neurons_network, neurons, d_neurons)
            d_weights = d_weights + d_weights_dummy

        end do

    end subroutine get_sens







end module NN

program main

    implicit none
    write(*,*) "Hello World!!"

end program main
