function [CB, RCB] = ccq_decode(model, Qx, Qy)

    if isempty(Qx) && isempty(Qy) || ~isempty(Qx) && ~isempty(Qy)
        error('both input matrices are empty or unempty.');

    elseif ~isempty(Qx) && isempty(Qy)

        if (max(Qx(:)) == max(model.k))
            error('Qx should be zero-based.');
        end

        m = model.m;
        n = size(Qx, 2);

        CB = zeros(size(model.Cx{1}, 1), n);

        for i = 1:m
            CB = CB + model.Cx{i}(:, 1 + uint16(Qx(i, :)));
        end

        RCB = model.Rx * CB;

    elseif isempty(Qx) && ~isempty(Qy)

        if (max(Qy(:)) == max(model.k))
            error('Qy should be zero-based.');
        end

        m = model.m;
        n = size(Qy, 2);

        CB = zeros(size(model.Cy{1}, 1), n);

        for i = 1:m
            CB = CB + model.Cy{i}(:, 1 + uint16(Qy(i, :)));
        end

        RCB = model.Ry * CB;
    end

end
