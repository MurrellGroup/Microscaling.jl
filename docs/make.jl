using Microscaling
using Documenter

DocMeta.setdocmeta!(Microscaling, :DocTestSetup, :(using Microscaling); recursive=true)

makedocs(;
    modules=[Microscaling],
    authors="Anton Oresten <antonoresten@proton.me> and contributors",
    sitename="Microscaling.jl",
    format=Documenter.HTML(;
        canonical="https://MurrellGroup.github.io/Microscaling.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/MurrellGroup/Microscaling.jl",
    devbranch="main",
)
