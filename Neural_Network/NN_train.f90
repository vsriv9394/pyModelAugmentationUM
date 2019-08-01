module mod_train

    use mod_sens_predict
    use mod_loss_functions
    use mod_optimization_algorithms

    implicit none
    contains

    subroutine train(neural_net, features, beta_target, batch_size, n_epochs, train_fraction, verbose)

    ! Function get_weights_sens
    ! ::::::::::::::::::::::::::
    ! Desc:  Get sensitivity for weights given the batch indices, features and beta sensitivities
    ! nargs: 6
    ! args:  neural_net     <type(neural_network),   input/output>    Neural Network
    !        features       <real(kind=8) 1-D array, input>           Features to be used as inputs during prediction
    !        beta_target    <real(kind=8),           input>           Target output for the neural network
    !        batch_size     <real(kind=8) 1-D array, output>          Batch size for batch gradient descent
    !        n_epochs       <integer,                input>           Number of epochs to run the training for
    !        train_fraction <integer,                input>           Initial fraction of the data used for training

        implicit none

        type(neural_network), intent(inout) :: neural_net
        real*8,               intent(in)    :: features(:,:), beta_target(:), train_fraction
        integer,              intent(in)    :: batch_size, n_epochs, verbose

    !-----------------------------------------------------------------------------------------------------------------------

        integer :: i_epoch, batch_start, batch_end
        real*8  :: beta(size(features, 2)), d_beta(size(features, 2)), d_weights(size(neural_net%weights)), loss

        ! Loop over epochs

        do i_epoch = 1, n_epochs

            batch_start = 1

            ! For every epoch loop over batches

            do
            
                batch_end = batch_start + batch_size

                ! Correct batch size in the final batch

                if (batch_end>nint(dble(size(features,2))*train_fraction)) then

                    batch_end = nint(dble(size(features,2))*train_fraction)

                end if          
                
                call predict(neural_net, features, beta)
                call apply_loss_function(neural_net%loss_fn_name, beta, beta_target, loss, d_beta)
                call get_weights_sens(neural_net, features, d_beta, d_weights, batch_start, batch_end)
                call optimizer_update(neural_net, d_weights)

                batch_start = batch_end + 1

                if (batch_start>nint(dble(size(features,2))*train_fraction)) then

                    exit

                end if          
                
            end do

            if (verbose==1 .or. (i_epoch==1 .or. i_epoch==n_epochs)) then
                write(*,"(A,I9,A27,ES15.6E2)") "Iteration ", i_epoch, "   Loss function ", loss
            end if

        end do

    end subroutine train

end module mod_train
