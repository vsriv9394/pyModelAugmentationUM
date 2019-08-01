module nn

    use mod_train
    implicit none
    contains

    subroutine nn_get_weights_sens(n_neurons, act_fn_name, loss_fn_name, opt_name, n_weights, weights, n_features, n_data, &
                                   features, batch_start, batch_end, d_beta, d_weights, opt_params)
    
        implicit none
    
        integer,           intent(in)  :: n_neurons(:), n_weights, n_features, n_data, batch_start, batch_end
        character(len=10), intent(in)  :: act_fn_name, loss_fn_name, opt_name
        real*8,            intent(in)  :: weights(n_weights), opt_params(:)
        real*8,            intent(in)  :: features(n_features,n_data)
        real*8,            intent(in)  :: d_beta(:)
        real*8,            intent(out) :: d_weights(n_weights)
    
        type(neural_network)           :: neural_net
    
        call create_neural_network(neural_net, n_neurons, act_fn_name, loss_fn_name, opt_name, weights, opt_params)
    
        call get_weights_sens(neural_net, features, d_beta, d_weights, batch_start, batch_end)
    
        call delete_neural_network(neural_net)
    
    end subroutine nn_get_weights_sens
    
    subroutine nn_predict(n_neurons, act_fn_name, loss_fn_name, opt_name, n_weights, weights, n_features, n_data, &
                          features, beta, opt_params)
    
        implicit none
    
        integer,           intent(in)  :: n_neurons(:), n_features, n_weights, n_data
        character(len=10), intent(in)  :: act_fn_name, loss_fn_name, opt_name
        real*8,            intent(in)  :: weights(n_weights), opt_params(:)
        real*8,            intent(in)  :: features(n_features,n_data)
        real*8,            intent(out) :: beta(n_data)
    
        type(neural_network)           :: neural_net
    
        call create_neural_network(neural_net, n_neurons, act_fn_name, loss_fn_name, opt_name, weights, opt_params)
    
        call predict(neural_net, features, beta)
    
        call delete_neural_network(neural_net)
    
    end subroutine nn_predict
    
    subroutine nn_train(n_neurons, act_fn_name, loss_fn_name, opt_name, n_weights, weights, n_features, n_data, &
                        features, beta_target, batch_size, n_epochs, train_fraction, opt_params, verbose)
    
        implicit none
    
        integer,           intent(in)     :: n_neurons(:), n_features, n_weights, n_data, batch_size, n_epochs, verbose
        character(len=10), intent(in)     :: act_fn_name, loss_fn_name, opt_name
        real*8,            intent(inout)  :: weights(n_weights)
        real*8,            intent(in)     :: features(n_features,n_data), train_fraction
        real*8,            intent(in)     :: beta_target(n_data), opt_params(:)
    
        type(neural_network)           :: neural_net
    
        call create_neural_network(neural_net, n_neurons, act_fn_name, loss_fn_name, opt_name, weights, opt_params)
    
        call train(neural_net, features, beta_target, batch_size, n_epochs, train_fraction, verbose)
    
        call delete_neural_network(neural_net)
    
    end subroutine nn_train

end module nn

program main

end program main
