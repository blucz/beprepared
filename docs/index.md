# Home

## Overview

beprepared is an easy and efficient way to prepare high quality image datasets for diffusion model fine-tuning.

It falicates both human and machine-driven data prep work in a non-destructive environment that aggressively 
avoids duplicated effort, even as your workflow evolves.

It is the most efficient way for one person to prepare image data sets with thousands of images or more.

## Code

The code is published on [GitHub](https://github.com/blucz/beprepared).

Pull requests are welcome.

## Installation

You can install beprepared from PyPI:

    $ pip install beprepared

## Development Philosophy

While beprepared is a technical product, user-experience is the primary focus. By providing a good user experience,
people will create better datasets and better models. This is a win for everyone.

beprepared is batteries-included software. While it's possible to define nodes in other PyPI packages, the first 
choice should be to merge new functionality into `blucz/beprepared`. Extensions are not on our roadmap. While extension 
ecosystems are powerful, they can also lead to confusing user experiences that are not beginner friendly, fragmentation, 
and difficulty discovering functionality. 
