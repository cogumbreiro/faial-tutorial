
# Try it yourself, using GitHub Actions

Faial is available in the GitHub Action Marketplace as
[`cogumbreiro/setup-faial`](https://github.com/marketplace/actions/setup-faial).

<img src="images/gh.png">

To make `faial-drf` available in the `PATH`, simply add a step that uses
`cogumbreiro/setup-faial@v1.0`. Here's an example of how we setup Faial in
this repository to automatically check that `saxpy.cu` is data-race free.

```yaml
on: [push]

jobs:
  check_drf:
    runs-on: ubuntu-latest
    name: Check that saxpy.cu is DRF.
    steps:
    - name: Check out code
      uses: actions/checkout@v1
    - name: Setup faial
      uses: cogumbreiro/setup-faial@v1.0
    - name: Check saxpy
      run: |
        faial-drf saxpy.cu
```
