function [feature_value, new_category_label] = transfer_category_feature(feature_cell, category_label)
unique_label = unique(feature_cell);
add_label = (setdiff(unique_label, category_label))';
new_category_label = [category_label add_label];

%feature_value = zeros(length(feature_cell),1);

[Lia, feature_value] = ismember(feature_cell, new_category_label);

% for i = 1 : 1 : length(new_category_label)
%     ind = ismember(feature_cell,new_category_label(i));
%     feature_value(ind) = i;
% end
