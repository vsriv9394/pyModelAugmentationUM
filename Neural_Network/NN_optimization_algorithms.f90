module mod_optimization_algorithms

!======================================
! Optimization algorithms for learning
!======================================

    use mod_data_structures
    implicit none
    contains

    

    subroutine optimizer_update(neural_net, d_weights)

    ! Function optimizer_update
    ! ::::::::::::::::::::::::::
    ! Desc:  Update weights
    ! nargs: 2
    ! args:  neural_net   <type(neural_network),   input/output>    Neural Network
    !        d_weights    <real(kind=8) 1-D array, input>           Derivative of loss function w.r.t. weights

        implicit none

        type(neural_network), intent(inout) :: neural_net
        real*8,               intent(in)    :: d_weights(:)

    !-----------------------------------------------------------------------------------------------------------------------

        if (neural_net%opt_name=='adam') then

            call adam_optimizer_update(neural_net, d_weights)

        end if

    end subroutine optimizer_update

    
    
    
    
    subroutine adam_optimizer_update(neural_net, d_weights)

    ! Function optimizer_update
    ! ::::::::::::::::::::::::::
    ! Desc:  Update weights
    ! nargs: 2
    ! args:  neural_net   <type(neural_network),   input/output>    Neural Network
    !        d_weights    <real(kind=8) 1-D array, input>           Derivative of loss function w.r.t. weights

        implicit none

        type(neural_network), target, intent(inout) :: neural_net
        real*8,                       intent(in)    :: d_weights(:)

    !-----------------------------------------------------------------------------------------------------------------------

        real*8, pointer :: m(:), v(:), beta_1t, beta_2t
        real*8          :: alpha, beta_1, beta_2, eps

        m => neural_net%vectors(:,1)
        v => neural_net%vectors(:,2)

        alpha   = neural_net%params(1)
        beta_1  = neural_net%params(2)
        beta_2  = neural_net%params(3)
        eps     = neural_net%params(4)
        
        beta_1t => neural_net%params(5)
        beta_2t => neural_net%params(6)

        m = beta_1 * m + (1.0D0-beta_1) * d_weights
        v = beta_2 * v + (1.0D0-beta_2) * d_weights * d_weights

        beta_1t = beta_1t * beta_1
        beta_2t = beta_2t * beta_2

        neural_net%weights = neural_net%weights - alpha * sqrt(1.0D0 - beta_2t) / (1.0D0-beta_1t) * m / (sqrt(v) + eps)

    end subroutine adam_optimizer_update


end module mod_optimization_algorithms
