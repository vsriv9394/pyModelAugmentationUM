module mod_data_structures

    implicit none

    type neural_network
        character(len=10)    :: act_fn_name, loss_fn_name, opt_name
        integer              :: n_layers
        integer, pointer     :: n_neurons(:)
        real*8,  pointer     :: weights(:)
        real*8,  allocatable :: neurons(:)
        real*8,  allocatable :: vectors(:,:)
        real*8,  pointer     :: params(:)
    end type neural_network

    contains

    subroutine create_neural_network(neural_net, n_neurons, &
                                     act_fn_name, loss_fn_name, opt_name, &
                                     weights, opt_params)

    ! Function create_neural_network
    ! :::::::::::::::::::::::::::::::
    ! Desc:  Create a neural network
    ! nargs: 6
    ! args:  neural_net   <type(neural_network),   input/output>    Neural Network
    !        n_neurons    <integer 1-D array,      input>           Number of neurons n all layers
    !        act_fn_name  <character(len=10),      input>           Activation function name
    !        loss_fn_name <character(len=10),      input>           Loss function name
    !        opt_name     <character(len=10),      input>           Optimizer name
    !        weights      <real(kind=8) 1-D array, input>           Weights and biases provided for the neural network
    !        opt_params   <real(kind=8) 1-D array, input>           Parameters to tune optimizer

        implicit none

        type(neural_network), intent(out) :: neural_net
        integer, target,      intent(in)  :: n_neurons(:)
        character(len=10),    intent(in)  :: act_fn_name, loss_fn_name, opt_name
        real*8, target,       intent(in)  :: weights(:), opt_params(:)

    !-----------------------------------------------------------------------------------------------------------------------

        ! Set the names of optimizer, activation and loss function

        neural_net%act_fn_name  = act_fn_name
        neural_net%loss_fn_name = loss_fn_name
        neural_net%opt_name     = opt_name

        
        ! Set the number of layers, number of neurons and neurons

        neural_net%n_layers  = size(n_neurons)
        allocate(neural_net%neurons(sum(n_neurons)))
        neural_net%n_neurons => n_neurons

        
        ! Set the weights and optimizer parameters

        neural_net%weights   => weights
        neural_net%params    => opt_params
        neural_net%params(5) = 1.0D0
        neural_net%params(6) = 1.0D0

        
        ! Set the vectors to be used for different optimizers
        
        if (opt_name=='adam') then

            allocate(neural_net%vectors(size(weights),2))
            neural_net%vectors = 0.0D0

        else

            write(*,*) "Error (nn.mod_data_structures.create_neural_network): Specified optimizer not available. "//&
                       "Available options are:"
            write(*,*) "    - adam"
            stop

        end if

    end subroutine create_neural_network

    
    
    
    
    subroutine delete_neural_network(neural_net)

    ! Function delete_neural_network
    ! :::::::::::::::::::::::::::::::
    ! Desc:  Delete a neural network
    ! nargs: 1
    ! args:  neural_net   <type(neural_network),   input/output>    Neural Network

        implicit none

        type(neural_network), intent(inout) :: neural_net

    !-----------------------------------------------------------------------------------------------------------------------
        
        deallocate(neural_net%neurons)
        if (allocated(neural_net%vectors)) then
            deallocate(neural_net%vectors)
        end if
        nullify(neural_net%n_neurons)
        nullify(neural_net%weights)
        nullify(neural_net%params)

    end subroutine delete_neural_network

end module mod_data_structures
