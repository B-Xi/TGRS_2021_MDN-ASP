% rng ('default');
% rng(1);
load('./data/IN/Indian_pines_corrected.mat')
load('./data/IN/GRSM2018Standard/Indian_gt.mat')
load('./data/IN/GRSM2018Standard/IndianTR_gt.mat')
data = indian_pines_corrected;
image_gt = double(Indian_gt);
num_class=max(max(image_gt));
[rows,cols]=size(image_gt);
gt_flatten = reshape(image_gt,rows*cols,1);
trainingIndexRandom= find(IndianTR_gt_flatten~=0);
rndIDXt = randperm(length(trainingIndexRandom));
trainingIndexRandom = trainingIndexRandom(rndIDXt);
[trainingIndexRandom_rows,trainingIndexRandom_cols] = ind2sub(size(image_gt),trainingIndexRandom);
save IndianTrainingIndexRandom1percent.mat trainingIndexRandom trainingIndexRandom_rows trainingIndexRandom_cols
wholeSample = find(gt_flatten~=0);
testingIndexRandom=setdiff(wholeSample,trainingIndexRandom);
[testingIndexRandom_rows,testingIndexRandom_cols] = ind2sub(size(image_gt),testingIndexRandom);
save IndianTestingIndexRandom1percent.mat testingIndexRandom testingIndexRandom_rows testingIndexRandom_cols
%%
bb = gt_flatten(trainingIndexRandom);
cc=zeros(rows*cols,1);
cc(trainingIndexRandom)= bb;
mask_train =reshape(cc,rows,cols);
mask_all_test = image_gt-mask_train;
for j = 1:num_class
    mask_test = zeros(rows,cols);
    locationj = find(mask_all_test==j);
    mask_test(locationj) = mask_all_test (locationj);
    save(['mask_test_patch',num2str(j)],'mask_test')
%     eval(['save mask_test_patch', num2str(j), '.mat',mask_test])
end

save indian.mat data 
save mask_train_200_10 mask_train
numbers = zeros(num_class,1);
for i=1:num_class
    numbers(i,1) =size(find(gt_flatten==i),1);
end
