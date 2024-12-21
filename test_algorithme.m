%% test
%Téléchargement de l'image 
I = imread(sprintf('test_cadre.jpg'));
% Convertir en espace HSV
hsvImage = rgb2hsv(I);

% Séparer les canaux HSV
hue = hsvImage(:,:,1);        % Teinte
saturation = hsvImage(:,:,2); % Saturation
value = hsvImage(:,:,3);      % Valeur

% Définir les seuils pour détecter le bleu (ajuster si nécessaire)
hueThresholdLow = 0.55; % Borne basse de la teinte
hueThresholdHigh = 0.75; % Borne haute de la teinte
saturationThreshold = 0.3; % Seuil minimum pour la saturation
valueThreshold = 0.2; % Seuil minimum pour la luminosité

% Créer un masque binaire pour les pixels bleus
blueMask = (hue >= hueThresholdLow) & (hue <= hueThresholdHigh) & ...
           (saturation >= saturationThreshold) & ...
           (value >= valueThreshold);

% Optionnel : Nettoyer le masque avec des opérations morphologiques
blueMask = imopen(blueMask, strel('square', 3)); % Suppression de bruit
blueMask = imclose(blueMask, strel('square', 3)); % Remplir les trous

% Créer l'image en noir et blanc
outputImage = double(blueMask) * 255; % Convertir le masque en image 8 bits

% % Afficher les résultats
% figure;
% subplot(1, 2, 1);
% imshow(I);
% title('Image originale');
% 
% subplot(1, 2, 2);
% imshow(outputImage);
% title('Carrés bleus détectés (en blanc)');



%Mettre en evidence les carrées bleu 
Ig = I(:,:,3) > 180; % application du filtre pour ne garder que le carré bleu

Ig=double(Ig);

% etiquetage région 4 connexité , Lb contient les regions bleu
[Ir,numb]=bwlabel(outputImage); 

s=zeros(numb,1); %stocker la taille 
c=zeros(numb,2); %pour stocker les coordonnées x,y

imshow(outputImage)

for i=1:numb 
    [y,x ]=find(Ir==i);
    c(i,:)=[mean(x),mean(y)];
    s(i)=length(x);
end

hold on;
plot(c(:,1),c(:,2),'or','LineWidth',3);
c

%% fin


%Mesurre des coordonnées des points en pixels et en mm

p1 = [0 0; 0 10; 0 5; 5 0; 5 10; 10 0; 10 10; 10 5]*10; % [mm]10
p2 = [535.6815 524.6222; 354.9758 668.9485; 456.1223 588.6277; 676.1259  548.7203; 522.7450 708.2349; 830.5198 571.5254; 700.8333 743.8407; 774.8374 646.5122];

%Extractionn du nombre de points
n = size(p1,1);

%Initialisation de la matrice H 
A = zeros(2*n, 9);

%Remplissage de la matrice A 
for i = 1:n
    x = p1(i,1);
    y = p1(i,2);
    xp = p2(i,1);
    yp = p2(i,2);

    %A(2*i-1,:) = [0 0 0 -x -y -1 x*yp y*yp yp];
    %A(2*i,:) = [x y 1 0 0 0 -x*xp -y*xp -xp];

    A(2*i,:) = [0 0 0 x y 1 -x*yp -y*yp -yp];
    A(2*i-1,:) = [x y 1 0 0 0 -x*xp -y*xp -xp];
    
end

%Utilisation de la méthode DLT 
[U,S,V] = svd(A);
h = V(:,end);
H = reshape(h,[3 3])';
 
 

figure(1); % Create a new figure window
imshow(I); % Display the image
hold on;

%Méthode de hadriel sans stockage de doonnées 
%Calcul des coordonnées des nouveaux points et les afficher 
% for i = 1:n
%     xr = p1(i,:);
%     xpt = H*[p1(i,:) 1]'*21;
%     xpt = xpt/xpt(3);
%     plot(xpt(1),xpt(2),'ro')
% end

%Ma méthode avec sauvegarde de données 
%Calcul des coordonnées des nouveaux points et les afficher 

xpt1=zeros(n,3);
for i = 1:n
    xpt1(i,:) = H*[p1(i,:) 1]'*21;
    xpt1(i,:) = xpt1(i,:)/xpt1(i,3);
end

x = xpt1(:, 1); % Extraire les coordonnées X
y = xpt1(:, 2); % Extraire les coordonnées Y

plot(x, y, 'ro'); % 'b' pour bleu et 'o' pour des marqueurs circulaires
xlabel('X');
ylabel('Y');
title("Affichage de l'image avec les points respectifs");