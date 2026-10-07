function ret = RPE(Y_true, Y_predicted)

    % Compute the Relative Predictions Error

    ret = sum( (Y_true - Y_predicted).^2 ) / sum( Y_true.^2 );

end