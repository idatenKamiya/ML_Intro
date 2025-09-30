function [X_poly] = generate_poly_features(X, k)
    [m, n] = size(X);
    X_poly = zeros(size(X));  % Initialize empty matrix
    
    for root = 1:k
        X_root = sign(X) .* abs(X).^(1 / root);
        X_poly = [X_poly, X_root];  % Append as new columns
    end
end
