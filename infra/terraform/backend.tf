# Local backend. State file lives at terraform.tfstate (gitignored as plaintext);
# the infra/tofu wrapper sops+age-encrypts it to terraform.tfstate.enc on exit.
# See infra/README.md "State encryption model".
terraform {
  backend "local" {
    path = "terraform.tfstate"
  }
}
