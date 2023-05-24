from django.contrib import admin
from django.urls import path
from home import views

urlpatterns = [
    path("",views.index,name="home"),
    path("Login",views.Login,name="Login"),
    path("about",views.about,name="about"),
    path("fertilizer",views.fertilizer,name="fertilizer"),
    path('predict_crop', views.predict_crop,name="predict_crop"),
    path('predict_fertilizer', views.predict_fertilizer,name="predict_fertilizer"),
    path("contact",views.contact,name="contact"),
    path("validate_for_crop",views.validate_for_crop,name="validate_for_crop"),
    
    



]
