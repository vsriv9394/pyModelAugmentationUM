module mod_sens_predict

    use mod_propagate

    implicit none
    contains

    subroutine get_weights_sens(neural_net, features, d_beta, d_weights, batch_start, batch_end)

    ! Function get_weights_sens
    ! ::::::::::::::::::::::::::
    ! Desc:  Get sensitivity for weights given the batch indices, features and beta sensitivities
    ! nargs: 6
    ! args:  neural_net   <type(neural_network),   input/output>    Neural Network
    !        features     <real(kind=8) 1-D array, input>           Features to be used as inputs during prediction
    !        d_beta       <real(kind=8),           input>           Loss function derivatives
    !        d_weights    <real(kind=8) 1-D array, output>          Weight sensitivities to loss function
    !        batch_start  <integer,                input>           Starting index of the batch
    !        batch_end    <integer,                input>           Ending index of the batch

        implicit none

        type(neural_network), intent(inout) :: neural_net
        integer,              intent(in)    :: batch_start, batch_end
        real*8,               intent(in)    :: features(:,:)
        real*8,               intent(in)    :: d_beta(:)
        real*8,               intent(out)   :: d_weights(:)

    !-----------------------------------------------------------------------------------------------------------------------

        real*8  :: d_weights_dummy(size(d_weights))
        integer :: i_data

        if (batch_start<1) then
            write(*,*) "Error: Batch index starts from 1 (Had to choose from python and fortran conventions :P)"
            stop
        end if

        if (batch_end>size(features,2)) then
            write(*,*) "Error: Batch index exceeds number of input data points"
            stop
        end if

        d_weights = 0.0

        do i_data = batch_start, batch_end 

            call forwbackprop(neural_net, features(:,i_data), d_beta(i_data), d_weights_dummy)

            d_weights = d_weights + d_weights_dummy

        end do

    end subroutine get_weights_sens





    subroutine predict(neural_net, features, beta)

    ! Function get_weights_sens
    ! ::::::::::::::::::::::::::
    ! Desc:  Predict for the given features
    ! nargs: 6
    ! args:  neural_net   <type(neural_network),   input/output>    Neural Network
    !        features     <real(kind=8) 1-D array, input>           Features to be used as inputs during prediction
    !        beta         <real(kind=8),           output>           Prediction from the neural network

        implicit none
        
        type(neural_network), intent(inout) :: neural_net
        real*8,               intent(in)    :: features(:,:)
        real*8,               intent(out)   :: beta(:)

    !-----------------------------------------------------------------------------------------------------------------------

        integer :: i_data

        do i_data = 1, size(features, 2)

            call forwprop(neural_net, features(:,i_data), beta(i_data))

        end do

    end subroutine predict

end module mod_sens_predict
