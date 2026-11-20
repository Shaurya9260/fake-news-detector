pipeline {
    agent { label 'docker-lab' }

    options {
        skipDefaultCheckout(true)
        disableConcurrentBuilds()
    }
    triggers { pollSCM('H/2 * * * *') }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
                script {
                    def commit = bat(returnStdout: true, script: '@git rev-parse --short=12 HEAD').trim()
                    env.APP_TAG = "git-${commit}-${env.BUILD_NUMBER}"
                    env.CI_PROJECT = "fake-news-ci-${commit}-${env.BUILD_NUMBER}"
                }
            }
        }

        stage('Build image') {
            steps {
                bat 'docker version'
                bat 'docker compose version'
                bat 'docker build --tag fake-news-detector-web:%APP_TAG% .'
            }
        }

        stage('Test model and database') {
            environment { POSTGRES_PASSWORD = "${env.BUILD_TAG}-ci" }
            steps {
                bat 'docker compose -p %CI_PROJECT% up -d --wait --wait-timeout 180 db'
                bat 'docker compose -p %CI_PROJECT% run --rm --no-deps --pull never -T web python -m unittest discover -s tests -v'
            }
            post {
                always {
                    // This project owns only the disposable test database.
                    bat 'docker compose -p %CI_PROJECT% down --volumes --remove-orphans'
                }
            }
        }

        stage('Deploy and verify') {
            environment { POSTGRES_PASSWORD = credentials('fake-news-db-password') }
            steps {
                bat 'docker compose -p fake-news-detector up -d --no-build --pull never --wait --wait-timeout 180'
                bat '''docker compose -p fake-news-detector exec -T web python -c "from database import recent_predictions; recent_predictions(1); print('Web to PostgreSQL OK')"'''
                bat 'docker compose -p fake-news-detector ps'
            }
        }
    }
}
