# Image detection with Cloud Vision API.

## Introduction

This demo uses Google Cloud Image Recognition to identify images.
By passing a folder parameter extracts JPG or PNG images and contacts
Image Recognition API service, extracts entities and writes into a file
with the file name and the objects recognized.

Implements ThreadPoolExecutor to handle concurrent requests to API.

## Architecture

 - **File Reader** File Reader, list from a folder. 
 - **ExtractLabels** Extracts objects from an image via Cloud Vision. 

## Jupyter book

Extracts objects from Diego Rivera artwork and creates a matplotlib
graph with Top 25 objects from his art.

## Installation

Download apiclient library from Google.

## Customization

Modify read folder and API key.

[Google Cloud API][https://cloud.google.com/vision/]
