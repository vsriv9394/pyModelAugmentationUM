module mod_activation_functions

!================================================
! Module for neural network activation functions
!================================================
! Consider a value 'u' which has to be activated to a value 'a'
! and consider the derivative 'du/da' expressed purely in terms of 'a'

    implicit none
    contains
    
    subroutine apply_act_fn(act_fn_name, value)

    ! Function apply_act_fn
    ! ::::::::::::::::::::::
    ! Desc:  Apply activation function to a double precision variable and overwrite with result
    ! nargs: 2
    ! args:  act_fn_name <character(len=10), input>           Activation function name ("relu", "sigmoid")
    !        value       <real(kind=8),      input/output>    Input: u, Output: a

        implicit none

        character(len=10), intent(in)    :: act_fn_name
        real*8,            intent(inout) :: value

    ! ----------------------------------------------------------------------------------------------------------------------

        if (act_fn_name=='relu') then

            call relu(value)

        elseif (act_fn_name=='sigmoid') then

            call sigmoid(value)

        else

            write(*,*) "Error (nn.mod_activation.apply_act_fn): The specified activation function is not available. "//&
                       "Available choices are:"
            write(*,*) "    - relu"
            write(*,*) "    - sigmoid"
            stop

        end if

    end subroutine apply_act_fn

    
    
    
    
    subroutine apply_deriv_act_fn(act_fn_name, value)

    ! Function apply_deriv_act_fn
    ! ::::::::::::::::::::::::::::
    ! Desc:  Differentiate an activated value (act) w.r.t. its inactivated value (inact) in terms of the activated value
    !        and overwrite the activated value
    ! nargs: 2
    ! args:  act_fn_name <character(len=10), input>           Activation function name ("relu", "sigmoid")
    !        value       <real(kind=8),      input/output>    Input: a, Output: da/du

        implicit none

        real*8, intent(inout) :: value
        character(len=10), intent(in) :: act_fn_name

    ! ----------------------------------------------------------------------------------------------------------------------

        if (act_fn_name=='relu') then

            call d_relu(value)

        elseif (act_fn_name=='sigmoid') then

            call d_sigmoid(value)

        else

            write(*,*) "Error (nn.mod_activation.apply_act_fn): The specified activation function is not available. "//&
                       "Available choices are:"
            write(*,*) "    - relu"
            write(*,*) "    - sigmoid"
            stop

        end if

    end subroutine apply_deriv_act_fn





    subroutine relu(value)

    ! Function relu
    ! ::::::::::::::
    ! Desc:  Evaluate and overwrite with the relu activation
    ! nargs: 1
    ! args:  value       <real(kind=8),      input/output>    Input: u, Output: a = relu(u)

        implicit none

        real*8, intent(inout) :: value

    ! ----------------------------------------------------------------------------------------------------------------------

        if (value<0.0D0) then
        
            value = 0.0D0
                
        end if

    end subroutine relu





    subroutine d_relu(value)

    ! Function d_relu
    ! ::::::::::::::::
    ! Desc:  Differentiate relu activation w.r.t. inactivated value and overwrite
    ! nargs: 1
    ! args:  value       <real(kind=8),      input/output>    Input: a, Output: da/du = d_relu(a)

        implicit none

        real*8, intent(inout) :: value

    ! ----------------------------------------------------------------------------------------------------------------------

        if (value>0.0D0) then
        
            value = 1.0D0
                
        end if

    end subroutine d_relu





    subroutine sigmoid(value)

    ! Function sigmoid
    ! :::::::::::::::::
    ! Desc:  Evaluate and overwrite with the sigmoid activation
    ! nargs: 1
    ! args:  value       <real(kind=8),      input/output>    Input: u, Output: a = sigmoid(u)

        implicit none

        real*8, intent(inout) :: value

    ! ----------------------------------------------------------------------------------------------------------------------

        value = 1.0D0 / (1.0D0 + exp(-value))

    end subroutine sigmoid





    subroutine d_sigmoid(value)

    ! Function d_sigmoid
    ! :::::::::::::::::::
    ! Desc:  Differentiate sigmoid activation w.r.t. inactivated value and overwrite
    ! nargs: 1
    ! args:  value       <real(kind=8),      input/output>    Input: a, Output: da/du = d_sigmoid(a)

        implicit none

        real*8, intent(inout) :: value

    ! ----------------------------------------------------------------------------------------------------------------------

        value = value * (1.0D0 - value)

    end subroutine d_sigmoid


    subroutine unit_testing_activation_functions()

        implicit none

        real*8 :: a, b, c, d
        character(len=10) :: relu_name, sigmoid_name

        relu_name = 'relu'
        sigmoid_name = 'sigmoid'

        a =  2.3D0
        b = -1.2D0
        c =  0.5D0
        d = -0.25D0

        write(*,'(A)', advance="no") "Checking relu on positive values... "
        call apply_act_fn(relu_name,a)
        if (abs(a-2.3D0)<1.0D-14) then
            write(*,*) "PASSED"
        else
            write(*,*) "FAILED"
            stop
        end if

        write(*,'(A)', advance="no") "Checking relu on negative values... "
        call apply_act_fn(relu_name,b)
        if (abs(b-0.0D0)<1.0D-14) then
            write(*,*) "PASSED"
        else
            write(*,*) "FAILED"
            stop
        end if

        write(*,'(A)', advance="no") "Checking sigmoid on positive values... "
        call apply_act_fn(sigmoid_name,c)
        if (abs(c-0.62245933120185459D0)<1.0D-14) then
            write(*,*) "PASSED"
        else
            write(*,*) "FAILED"
            stop
        end if
        
        write(*,'(A)', advance="no") "Checking sigmoid on negative values... "
        call apply_act_fn(sigmoid_name,d)
        if (abs(d-0.43782349911420193D0)<1.0D-14) then
            write(*,*) "PASSED"
        else
            write(*,*) "FAILED"
            stop
        end if

        
        write(*,'(A)', advance="no") "Checking relu derivative on positive values... "
        call apply_deriv_act_fn(relu_name, a)
        if (abs(a-1.0D0)<1.0D-14) then
            write(*,*) "PASSED"
        else
            write(*,*) "FAILED"
            stop
        end if
        
        write(*,'(A)', advance="no") "Checking relu derivative on negative values... "
        call apply_deriv_act_fn(relu_name, b)
        if (abs(b-0.0D0)<1.0D-14) then
            write(*,*) "PASSED"
        else
            write(*,*) "FAILED"
            stop
        end if
        
        write(*,'(A)', advance="no") "Checking sigmoid derivative on positive values... "
        call apply_deriv_act_fn(sigmoid_name, c)
        if (abs(c-0.2350037122015945D0)<1.0D-14) then
            write(*,*) "PASSED"
        else
            write(*,*) "FAILED"
            stop
        end if
        
        write(*,'(A)', advance="no") "Checking sigmoid derivative on negative values... "
        call apply_deriv_act_fn(sigmoid_name, d)
        if (abs(d-0.24613408273759835D0)<1.0D-14) then
            write(*,*) "PASSED"
        else
            write(*,*) "FAILED"
            stop
        end if

    end subroutine unit_testing_activation_functions

end module mod_activation_functions
