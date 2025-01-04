import requests
import time
import pandas as pd
from datetime import datetime
from typing import Dict, Optional


class GitHubDataCollector:
    def __init__(self, token: str):
        self.headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        self.base_url = 'https://api.github.com'
        self.session = requests.Session()
        self.session.headers.update(self.headers)

        # Define fields that match the CSV structure
        self.fields = [
            'name', 'full_name', 'size', 'stargazers_count', 'language',
            'has_issues', 'has_pages', 'has_discussions', 'archived',
            'disabled', 'open_issues_count', 'allow_forking', 'forks',
            'open_issues', 'watchers', 'commits_count', 'contributors_count',
            'languages_count', 'readme_length'
        ]

    def _make_request(self, url: str, params: Optional[Dict] = None) -> Dict:
        try:
            response = self.session.get(url, params=params)
            remaining = int(response.headers.get('X-RateLimit-Remaining', 30))

            if remaining < 5:  # When nearing the limit
                reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                sleep_time = reset_time - time.time() + 1
                print(f"Search API limit reached. Sleeping for {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
                return self._make_request(url, params)

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Error making request to {url}: {str(e)}")
            raise

    def collect_repositories(self, base_query: str, max_repos: int = 5000) -> pd.DataFrame:
        search_url = f"{self.base_url}/search/repositories"
        repositories = []
        start_date = datetime(2014, 1, 1)
        end_date = datetime.now()

        while len(repositories) < max_repos and start_date < end_date:
            current_end = start_date + pd.Timedelta(days=30)  # Search in 1-month chunks
            if current_end > end_date:
                current_end = end_date

            date_range = f"created:{start_date.strftime('%Y-%m-%d')}..{current_end.strftime('%Y-%m-%d')}"
            query = f"{base_query} {date_range}"

            params = {
                'q': query,
                'sort': 'stars',
                'order': 'desc',
                'per_page': 100
            }

            page = 1
            while len(repositories) < max_repos:
                params['page'] = page
                print(f"Fetching page {page} for date range {date_range}")
                try:
                    response = self._make_request(search_url, params)
                    if not response.get('items'):
                        break

                    for repo in response['items']:
                        repositories.append({
                            'name': repo.get('name'),
                            'full_name': repo.get('full_name'),
                            'stargazers_count': repo.get('stargazers_count'),
                            'language': repo.get('language'),
                            'created_at': repo.get('created_at'),
                        })
                        if len(repositories) >= max_repos:
                            break

                    page += 1
                    if page > 10:  # Stop after 10 pages to avoid exceeding 1,000 results
                        break

                except Exception as e:
                    print(f"Error: {e}")
                    break

            start_date = current_end + pd.Timedelta(days=1)  # Move to the next date range

        return pd.DataFrame(repositories)


def main():
    github_token = 'PASTER_YOUR_TOKEN_HERE'
    if not github_token:
        raise ValueError("Please set the GITHUB_TOKEN environment variable")

    collector = GitHubDataCollector(github_token)

    # Query for public repositories with at least 1000 stars
    query = 'is:public stars:>=100'

    print(f"\nCollecting repositories for query: {query}")
    final_df = collector.collect_repositories(query, max_repos=5000)

    # Save to CSV
    final_df.to_csv('github_repositories.csv', index=False)
    print(f"\nCollected data for {len(final_df)} repositories")


if __name__ == "__main__":
    main()
