clc
clear all
close all
syms x y z t Re real
u = (sin(z) + cos(y))*exp(-t/Re);
v = (sin(x) + cos(z))*exp(-t/Re);
w = (sin(y) + cos(x))*exp(-t/Re);
p = (-sin(x)*cos(z) - cos(x)*sin(y) - cos(y)*sin(z);
U = [u,v,w]';
grad_U = [diff(u,x) diff(u,y) diff(u,z);diff(v,x) diff(v,y) diff(v,z); diff(w,x) diff(w,y) diff(w,z)];
Lapl_U = [diff(u,x,2) + diff(u,y,2) + diff(u,z,2);diff(v,x,2) + diff(v,y,2) + diff(v,z,2); diff(w,x,2) + diff(w,y,2) + diff(w,z,2)];
grad_P = [diff(p,x);diff(p,y);diff(p,z)];
partialU_partialt = [diff(u,t);diff(v,t); diff(w,t)];
div_U = diff(u,x) + diff(v,y) + diff(w,z)
strong_form = partialU_partialt + grad_U*U - 1/Re*Lapl_U - U/Re;
simplify(strong_form)
