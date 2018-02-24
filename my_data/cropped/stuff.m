%% Прочитать из папки все .bmp картинки в cell-array
dirlist = dir('sized/*.bmp');
n = size(dirlist,1);
clear C;
for i = 1:n
    C{i} = fullfile(dirlist(i).folder, dirlist(i).name);
end

%%
imnum = 1;
img = imread(C{imnum});
imtool(img);

%%
minval = min(img(:));
img2 = img - minval;

img2 = im2double(img2);
maxval = max(img2(:));
img2 = img2 / maxval;
