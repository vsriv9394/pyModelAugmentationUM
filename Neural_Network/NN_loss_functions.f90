module mod_loss_functions

!==========================================
! Module for neural network loss functions
!==========================================

    implicit none
    contains

    subroutine apply_loss_function(loss_fn_name, beta, beta_target, loss, loss_deriv)

    ! Function apply_loss_function
    ! :::::::::::::::::::::::::::::
    ! Desc:  Evaluate loss function and derivatives
    ! nargs: 5
    ! args:  loss_fn_name <character(len=10),      input>           Activation function name ("relu", "sigmoid")
    !        beta         <real(kind=8) 1-D array, input>           Neural Network Prediction
    !        beta_target  <real(kind=8) 1-D array, input>           Target Output (or Labels) for learning
    !        loss         <real(kind=8),           output>          Loss function value
    !        loss_deriv   <real(kind=8) 1-D array, output>          Derivatives of loss w.r.t. beta

        implicit none

        real*8, intent(in)            :: beta(:), beta_target(:)
        real*8, intent(out)           :: loss, loss_deriv(size(beta))
        character(len=10), intent(in) :: loss_fn_name

    !-----------------------------------------------------------------------------------------------------------------------

        if (size(beta)/=size(beta_target)) then

            write(*,*) "Error (nn.mod_loss_functions.apply_loss_function): beta and beta_target should have the same size"
            stop

        end if

        if (loss_fn_name=='mse') then

            call mean_squared_error(beta, beta_target, loss, loss_deriv)

        else

            write(*,*) "Error (nn.mod_loss_functions.apply_loss_function): Specified loss function not available. "//&
                       "Available options are:"
            write(*,*) "    - mse (mean squared error loss)"
            stop

        end if

    end subroutine apply_loss_function





    subroutine mean_squared_error(beta, beta_target, loss, loss_deriv)

    ! Function apply_loss_function
    ! :::::::::::::::::::::::::::::
    ! Desc:  Evaluate mean squared loss function and derivatives
    ! nargs: 4
    ! args:  beta         <real(kind=8) 1-D array, input>           Neural Network Prediction
    !        beta_target  <real(kind=8) 1-D array, input>           Target Output (or Labels) for learning
    !        loss         <real(kind=8),           output>          Loss function value
    !        loss_deriv   <real(kind=8) 1-D array, output>          Derivatives of loss w.r.t. beta

        implicit none

        real*8, intent(in)  :: beta(:), beta_target(:)
        real*8, intent(out) :: loss, loss_deriv(size(beta))
        
    !-----------------------------------------------------------------------------------------------------------------------

        integer :: i_data

        loss = sum((beta-beta_target)**2) / dble(size(beta))
        loss_deriv = 2.0D0 * (beta - beta_target) / dble(size(beta))

    end subroutine mean_squared_error


end module mod_loss_functions
