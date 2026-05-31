"""
cloud_run_deploy.py — Deploy Atlas to Google Cloud Run.

Cloud Run is the right target for stateless agent API deployments:
  - Scales to zero when idle (no cost when not processing)
  - Scales to hundreds of instances under load
  - No VMs, no Kubernetes, no maintenance

This script handles the full deployment pipeline:
  1. Build Docker image
  2. Push to Artifact Registry
  3. Deploy to Cloud Run with agent-appropriate settings
     (higher memory, longer request timeout for long-running tasks)

Usage:
    python cloud_run_deploy.py --project my-project --region us-central1
    python cloud_run_deploy.py --project my-project --tag v0.19.1

Requires: pip install google-cloud-run google-cloud-artifact-registry
          gcloud CLI installed and authenticated
"""

import argparse
import os
import subprocess
import sys

# ── Configuration ─────────────────────────────────────────────────────

DEFAULTS = {
    "project":        os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id"),
    "region":         "us-central1",
    "service":        "atlas-agent-api",
    "image_name":     "atlas-agent",
    "tag":            "latest",
    "memory":         "2Gi",        # Agent workloads need more than default 512Mi
    "cpu":            "2",          # 2 vCPUs for concurrent tool calls
    "concurrency":    "10",         # Max requests per instance
    "timeout":        "300",        # 5 min timeout for long-running tasks
    "min_instances":  "0",          # Scale to zero
    "max_instances":  "100",
}


def run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command with visible output."""
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, text=True, capture_output=False)
    if check and result.returncode != 0:
        print(f"  Error: command failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)
    return result


def build_and_push(project: str, region: str, image_name: str, tag: str) -> str:
    """Build the Docker image and push to Artifact Registry."""
    registry  = f"{region}-docker.pkg.dev"
    repo      = f"{registry}/{project}/atlas"
    image_uri = f"{repo}/{image_name}:{tag}"

    print("\n1. Building Docker image...")
    run(f"docker build -t {image_uri} .")

    print("\n2. Pushing to Artifact Registry...")
    # Ensure Artifact Registry repo exists
    run(
        f"gcloud artifacts repositories create atlas "
        f"--repository-format=docker --location={region} "
        f"--project={project} --quiet",
        check=False,  # OK if repo already exists
    )
    run(f"gcloud auth configure-docker {registry} --quiet")
    run(f"docker push {image_uri}")

    return image_uri


def deploy(project: str, region: str, service: str, image_uri: str, cfg: dict) -> str:
    """Deploy the image to Cloud Run."""
    print("\n3. Deploying to Cloud Run...")

    env_vars = ",".join([
        "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY",
        "REDIS_URL=redis://10.x.x.x:6379/0",  # Replace with your Memorystore URL
    ])

    cmd = (
        f"gcloud run deploy {service} "
        f"--image={image_uri} "
        f"--project={project} "
        f"--region={region} "
        f"--platform=managed "
        f"--memory={cfg['memory']} "
        f"--cpu={cfg['cpu']} "
        f"--concurrency={cfg['concurrency']} "
        f"--timeout={cfg['timeout']} "
        f"--min-instances={cfg['min_instances']} "
        f"--max-instances={cfg['max_instances']} "
        f"--set-env-vars={env_vars} "
        f"--allow-unauthenticated "  # Remove for internal services
        f"--quiet"
    )
    run(cmd)

    # Get the service URL
    result = subprocess.run(
        f"gcloud run services describe {service} "
        f"--region={region} --project={project} "
        f"--format='value(status.url)'",
        shell=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description="Deploy Atlas Agent to Cloud Run")
    parser.add_argument("--project",    default=DEFAULTS["project"])
    parser.add_argument("--region",     default=DEFAULTS["region"])
    parser.add_argument("--service",    default=DEFAULTS["service"])
    parser.add_argument("--image-name", default=DEFAULTS["image_name"])
    parser.add_argument("--tag",        default=DEFAULTS["tag"])
    parser.add_argument("--memory",     default=DEFAULTS["memory"])
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip build+push and re-deploy existing image")
    args = parser.parse_args()

    cfg = {k: getattr(args, k, DEFAULTS[k]) for k in DEFAULTS}

    print(f"Deploying Atlas v0.19 to Cloud Run")
    print(f"  Project:  {args.project}")
    print(f"  Region:   {args.region}")
    print(f"  Service:  {args.service}")
    print(f"  Memory:   {args.memory}  CPU: {DEFAULTS['cpu']}")

    if args.skip_build:
        registry  = f"{args.region}-docker.pkg.dev"
        image_uri = f"{registry}/{args.project}/atlas/{args.image_name}:{args.tag}"
    else:
        image_uri = build_and_push(args.project, args.region, args.image_name, args.tag)

    service_url = deploy(args.project, args.region, args.service, image_uri, cfg)

    print(f"\nDeployment complete!")
    print(f"  Service URL: {service_url}")
    print(f"\nTest it:")
    print(f'  curl -X POST {service_url}/agent/run \\')
    print(f'    -H "Content-Type: application/json" \\')
    print(f'    -d \'{{"message": "What is RAG?", "session_id": "test-1"}}\'')


if __name__ == "__main__":
    main()
