from django import forms
from .models import ImageAnalysis


class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = ImageAnalysis
        fields = ["title", "image"]
        widgets = {
            "title": forms.TextInput(attrs={
                "placeholder": "Analysis name (optional)",
                "class": "form-input",
            }),
            "image": forms.FileInput(attrs={
                "accept": "image/*",
                "class": "form-file",
            }),
        }