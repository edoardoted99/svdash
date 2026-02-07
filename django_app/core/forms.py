from django import forms
from .models import ImageAnalysis


class ImageUploadForm(forms.ModelForm):
    grayscale = forms.BooleanField(
        required=False,
        initial=False,
        widget=forms.CheckboxInput(attrs={"class": "form-checkbox", "id": "id_grayscale"}),
    )

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
