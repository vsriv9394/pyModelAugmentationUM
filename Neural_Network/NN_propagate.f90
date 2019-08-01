module mod_propagate

! Propagate functions for the neural network for one data point

    use mod_activation_functions
    use mod_data_structures
    implicit none
    contains

    subroutine forwprop(neural_net, features, beta)

    ! Function forwprop
    ! ::::::::::::::::::
    ! Desc:  Forward Propagation
    ! nargs: 3
    ! args:  neural_net   <type(neural_network),   input/output>    Neural Network
    !        features     <real(kind=8) 1-D array, input>           Features to be used as inputs during prediction
    !        beta         <real(kind=8),           output>          Value predicted by the neural network

        implicit none
        
        type(neural_network), intent(inout) :: neural_net
        real*8,               intent(in)    :: features(:)
        real*8,               intent(out)   :: beta

    ! ----------------------------------------------------------------------------------------------------------------------

        integer :: neuron_start, neuron_end, bias, weight_start, weight_end
        integer :: i_layer, i_neuron

        ! neuron_start : Network index of first neuron of the previous layer
        ! neuron_end   : Network index of last neuron of the previous layer
        ! i_layer      : Index of the current layer
        ! i_neuron     : Layer index of current neuron ( Network index = neuron_end + i_neuron )
        ! bias         : Network index of bias for the current neuron
        ! weight_start : Network index of weight corresponding to the first neuron in previous layer for current neuron
        ! weight_end   : Network index of weight corresponding to the last neuron in previous layer for current neuron
        
        ! Set input neurons to have values as features
        
        neural_net%neurons(1:neural_net%n_neurons(1)) = features

        ! Initialize weight_end and neuron_end to 0

        weight_end = 0
        neuron_end  = 0

        ! Loop over all layers except the input layer

        do i_layer = 2, neural_net%n_layers

            ! Set neuron_start and neuron_end as the neuron indices for the layer preceding this one by incrementing
            ! to neuron_end

            neuron_start = neuron_end + 1
            neuron_end   = neuron_end + neural_net%n_neurons(i_layer-1)

            ! Loop over all neurons in this layer

            do i_neuron = 1, neural_net%n_neurons(i_layer)

                ! Set bias index and weight_start and weight_end by incrementing to weight_end for previous neuron
                
                bias         = weight_end + 1
                weight_start = bias + 1
                weight_end   = bias + neural_net%n_neurons(i_layer-1)

                ! Set the value of this neuron to an affine transformation of previous layer

                neural_net%neurons(neuron_end + i_neuron) = neural_net%weights(bias) + &
                                                            sum(neural_net%weights(weight_start:weight_end) * &
                                                                neural_net%neurons(neuron_start:neuron_end))

                ! Apply activation function to this neuron
                
                if (i_layer<neural_net%n_layers) then
                
                    call apply_act_fn(neural_net%act_fn_name, neural_net%neurons(neuron_end + i_neuron))

                end if

            end do

        end do

        ! Set beta to the last neuron

        beta = neural_net%neurons(neuron_end+1)

    end subroutine forwprop





    subroutine forwbackprop(neural_net, features, d_beta, d_weights)

    ! Function forwprop
    ! ::::::::::::::::::
    ! Desc:  Forward and Backward Propagation
    ! nargs: 4
    ! args:  neural_net   <type(neural_network),   input/output>    Neural Network
    !        features     <real(kind=8) 1-D array, input>           Features to be used as inputs during prediction
    !        d_beta       <real(kind=8),           input>           Loss function derivatives
    !        d_weights    <real(kind=8) 1-D array, output>          Weight sensitivities to loss function

        implicit none
        
        type(neural_network), intent(inout) :: neural_net
        real*8,               intent(in)    :: d_beta, features(:)
        real*8,               intent(out)   :: d_weights(size(neural_net%weights))

    ! ----------------------------------------------------------------------------------------------------------------------

        integer :: neuron_start, neuron_end, bias, weight_start, weight_end
        integer :: i_layer, i_neuron
        real*8  :: d_neurons(size(neural_net%neurons))

        ! neuron_start : Network index of first neuron of the previous layer
        ! neuron_end   : Network index of last neuron of the previous layer
        ! i_layer      : Index of the current layer
        ! i_neuron     : Layer index of current neuron ( Network index = neuron_end + i_neuron )
        ! bias         : Network index of bias for the current neuron
        ! weight_start : Network index of weight corresponding to the first neuron in previous layer for current neuron
        ! weight_end   : Network index of weight corresponding to the last neuron in previous layer for current neuron
        
        ! Set input neurons to have values as features
        
        neural_net%neurons(1:neural_net%n_neurons(1)) = features

        ! Initialize weight_end and neuron_end to 0

        weight_end = 0
        neuron_end  = 0

        ! Loop over all layers except the input layer

        do i_layer = 2, neural_net%n_layers

            ! Set neuron_start and neuron_end as the neuron indices for the layer preceding this one by incrementing
            ! to neuron_end

            neuron_start = neuron_end + 1
            neuron_end   = neuron_end + neural_net%n_neurons(i_layer-1)

            ! Loop over all neurons in this layer

            do i_neuron = 1, neural_net%n_neurons(i_layer)

                ! Set bias index and weight_start and weight_end by incrementing to weight_end for previous neuron
                
                bias         = weight_end + 1
                weight_start = bias + 1
                weight_end   = bias + neural_net%n_neurons(i_layer-1)

                ! Set the value of this neuron to an affine transformation of previous layer

                neural_net%neurons(neuron_end + i_neuron) = neural_net%weights(bias) + &
                                                            sum(neural_net%weights(weight_start:weight_end) * &
                                                                neural_net%neurons(neuron_start:neuron_end))

                ! Apply activation function to this neuron
                
                if (i_layer<neural_net%n_layers) then
                
                    call apply_act_fn(neural_net%act_fn_name, neural_net%neurons(neuron_end + i_neuron))

                end if

            end do

        end do

    ! ----------------------------------------------------------------------------------------------------------------------

        ! Place the neuron_start at the neuron in the outer layer

        neuron_start = neuron_end + 1

        ! Place the bias index just outside the weights array

        bias         = weight_end + 1

        ! Place the d_beta at the last neuron

        d_neurons               = 0.0
        d_neurons(neuron_start) = d_beta

        ! Loop over the layers from the output layer to first hidden layer in the reverse order

        do i_layer = neural_net%n_layers, 2, -1

            ! For this layer, set neuron_start and neuron_end to the indices in the previous layer

            neuron_end   = neuron_start - 1
            neuron_start = neuron_start - neural_net%n_neurons(i_layer-1)

            ! Loop over all the neurons in this layer in the reverse order

            do i_neuron = neural_net%n_neurons(i_layer), 1, -1

                ! Set bias index, weight_start and weight_end to the ones for this neuron

                weight_end   = bias - 1
                weight_start = bias - neural_net%n_neurons(i_layer-1)
                bias         = weight_start - 1

                ! Multiply activation derivative to the derivative d_neuron for all neurons in the hidden layers

                if (i_layer < neural_net%n_layers) then
                
                    call apply_deriv_act_fn(neural_net%act_fn_name, neural_net%neurons(neuron_end + i_neuron))
                    
                    d_neurons(neuron_end+i_neuron) = d_neurons(neuron_end+i_neuron) * neural_net%neurons(neuron_end+i_neuron)
                
                end if

                ! Set bias and weight derivatives as is required

                d_weights(bias)                    = d_neurons(neuron_end+i_neuron)

                d_weights(weight_start:weight_end) = d_weights(bias) * neural_net%neurons(neuron_start:neuron_end)

                d_neurons(neuron_start:neuron_end) = d_neurons(neuron_start:neuron_end) + &
                                                     d_weights(bias) * neural_net%weights(weight_start:weight_end)

            end do

        end do

    end subroutine forwbackprop

end module mod_propagate