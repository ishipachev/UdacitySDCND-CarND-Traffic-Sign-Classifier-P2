dirname = 'cropped';
for i=3:17
    filename = C{i};
    img = imread(filename);
    outname = fullfile(dirname, filename);
    img_cropped = imresize(img, [32 32]);
    imwrite(img_cropped, outname);
end