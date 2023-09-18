FROM zivgitlab.wwu.io/pymor/pymor/ci-current:main

# binder wants to set the NB_ vars anyways, so we use it to service both setups
ARG NB_USER
ARG NB_UID

USER root
RUN rm -rf /pymor && useradd -d /pymor --shell /bin/bash -u ${NB_UID} -o -c "" -m ${NB_USER}
RUN pip install git+https://github.com/pymor/pymor@daab63b45688ba5282cb601982fd5181d3435e4e

ENV USER=${NB_USER} \
    HOME=/pymor

RUN cp -r * /pymor/
RUN cd /pymor && wget  https://docs.pymor.org/main/_downloads/049752ef291d62bd7bcde4a95068666e/tutorial_bt.ipynb https://docs.pymor.org/main/_downloads/b5c148b31f5986dd6e449df7623b0e08/tutorial_mor_with_anns.ipynb https://docs.pymor.org/main/_downloads/1000b3f568829899be8a4f7a0b7f7e9a/tutorial_optimization.ipynb https://docs.pymor.org/main/_downloads/711687a99e3da3e3418d3972ceb042e5/tutorial_builtin_discretizer.ipynb https://docs.pymor.org/main/_downloads/9457b171c152971cf5260fa1be5262db/tutorial_lti_systems.ipynb https://docs.pymor.org/main/_downloads/36d790e69b87fb8012d42aadb38afd2c/tutorial_external_solver.ipynb https://docs.pymor.org/main/_downloads/1f641b2cd53373bc47fc16373ad56a4f/tutorial_interact.ipynb https://docs.pymor.org/main/_downloads/e83e03d63417bf02b638d5214e9dc015/getting_started.ipynb https://docs.pymor.org/main/_downloads/218cda5bb1e3c69656498232df908d06/tutorial_unstable_lti_systems.ipynb https://docs.pymor.org/main/_downloads/242c69f4d668a8e7d0652f1c7a06f83b/tutorial_projection.ipynb https://docs.pymor.org/main/_downloads/1a27ade3bd8436eee3f8b145a3a5890b/tutorial_basis_generation.ipynb
RUN chown -R ${NB_USER} /pymor
USER ${NB_USER}

ENTRYPOINT []
WORKDIR /pymor
